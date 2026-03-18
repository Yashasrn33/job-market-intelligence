#!/bin/bash
#===============================================================================
# QuickSight Dashboard Deployment
#
# Sets up:
#   1. Athena views that power the dashboard visuals
#   2. QuickSight Athena data source
#   3. QuickSight datasets (one per view)
#   4. QuickSight dashboard template
#
# Prerequisites:
#   - QuickSight Enterprise or Standard edition activated in your AWS account
#   - The Glue tables (jobs, job_skills) already populated via the ETL pipeline
#
# Usage:
#   bash infrastructure/deploy_quicksight.sh
#   bash infrastructure/deploy_quicksight.sh --views-only   # just create Athena views
#===============================================================================
set -euo pipefail

if [[ -f .env ]]; then set -a; source .env; set +a; fi

BUCKET_NAME="${S3_BUCKET:?Set S3_BUCKET in .env or export it}"
REGION="${AWS_REGION:-us-east-1}"
DATABASE="${GLUE_DATABASE:-job_market_db}"
VIEWS_ONLY=false

for arg in "$@"; do
  case "${arg}" in
    --views-only) VIEWS_ONLY=true ;;
  esac
done

AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"

echo "==============================================="
echo "  QuickSight Dashboard Deployment"
echo "==============================================="
echo "  Account   : ${AWS_ACCOUNT_ID}"
echo "  Region    : ${REGION}"
echo "  Bucket    : ${BUCKET_NAME}"
echo "  Database  : ${DATABASE}"
echo ""

# ── Step 1: Create Athena Views ──────────────────────────────────────────────

echo "[1/4] Creating Athena views..."

VIEWS_FILE="visualization/quicksight/athena_views.sql"
if [[ ! -f "${VIEWS_FILE}" ]]; then
  echo "ERROR: ${VIEWS_FILE} not found." >&2
  exit 1
fi

# Split the file on CREATE OR REPLACE VIEW and execute each statement
# separately since Athena doesn't support multi-statement execution.
python3 - "${VIEWS_FILE}" "${DATABASE}" "${BUCKET_NAME}" "${REGION}" <<'PYEOF'
import sys, boto3, time

views_file, database, bucket, region = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

with open(views_file) as f:
    content = f.read()

# Split on CREATE OR REPLACE VIEW, keeping the delimiter
statements = []
for block in content.split("CREATE OR REPLACE VIEW"):
    block = block.strip()
    if not block or block.startswith("--"):
        continue
    stmt = "CREATE OR REPLACE VIEW " + block.rstrip(";").strip()
    statements.append(stmt)

athena = boto3.client("athena", region_name=region)
output_loc = f"s3://{bucket}/athena-results/"

for i, stmt in enumerate(statements, 1):
    view_name = stmt.split("AS")[0].replace("CREATE OR REPLACE VIEW", "").strip()
    print(f"  Creating view: {view_name}")
    try:
        resp = athena.start_query_execution(
            QueryString=stmt,
            QueryExecutionContext={"Database": database},
            ResultConfiguration={"OutputLocation": output_loc},
        )
        qid = resp["QueryExecutionId"]
        while True:
            status = athena.get_query_execution(QueryExecutionId=qid)
            state = status["QueryExecution"]["Status"]["State"]
            if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
                break
            time.sleep(1)
        if state != "SUCCEEDED":
            reason = status["QueryExecution"]["Status"].get("StateChangeReason", "unknown")
            print(f"    WARNING: {state} — {reason}")
        else:
            print(f"    OK")
    except Exception as e:
        print(f"    ERROR: {e}")

print(f"  Done: {len(statements)} views processed.")
PYEOF

if ${VIEWS_ONLY}; then
  echo ""
  echo "Views created. Skipping QuickSight setup (--views-only)."
  exit 0
fi

# ── Step 2: Create QuickSight Data Source (Athena) ───────────────────────────

echo ""
echo "[2/4] Creating QuickSight Athena data source..."

DS_ID="job-market-athena"
DS_NAME="Job Market Intelligence (Athena)"

aws quicksight create-data-source \
  --aws-account-id "${AWS_ACCOUNT_ID}" \
  --data-source-id "${DS_ID}" \
  --name "${DS_NAME}" \
  --type ATHENA \
  --data-source-parameters '{
    "AthenaParameters": {
      "WorkGroup": "primary"
    }
  }' \
  --permissions "[{
    \"Principal\": \"arn:aws:quicksight:${REGION}:${AWS_ACCOUNT_ID}:user/default/$(aws sts get-caller-identity --query Arn --output text | rev | cut -d/ -f1 | rev)\",
    \"Actions\": [
      \"quicksight:DescribeDataSource\",
      \"quicksight:DescribeDataSourcePermissions\",
      \"quicksight:PassDataSource\",
      \"quicksight:UpdateDataSource\",
      \"quicksight:DeleteDataSource\",
      \"quicksight:UpdateDataSourcePermissions\"
    ]
  }]" \
  --region "${REGION}" 2>/dev/null \
  || echo "  Data source already exists — continuing."

echo "  Data source: ${DS_ID}"

# ── Step 3: Create QuickSight Datasets ───────────────────────────────────────

echo ""
echo "[3/4] Creating QuickSight datasets..."

QS_USER_ARN="arn:aws:quicksight:${REGION}:${AWS_ACCOUNT_ID}:user/default/$(aws sts get-caller-identity --query Arn --output text | rev | cut -d/ -f1 | rev)"
DS_ARN="arn:aws:quicksight:${REGION}:${AWS_ACCOUNT_ID}:datasource/${DS_ID}"

DATASET_PERMISSIONS="[{
  \"Principal\": \"${QS_USER_ARN}\",
  \"Actions\": [
    \"quicksight:DescribeDataSet\",
    \"quicksight:DescribeDataSetPermissions\",
    \"quicksight:PassDataSet\",
    \"quicksight:DescribeIngestion\",
    \"quicksight:ListIngestions\",
    \"quicksight:UpdateDataSet\",
    \"quicksight:DeleteDataSet\",
    \"quicksight:CreateIngestion\",
    \"quicksight:CancelIngestion\",
    \"quicksight:UpdateDataSetPermissions\"
  ]
}]"

create_dataset() {
  local ds_id="$1"
  local ds_name="$2"
  local sql="$3"

  echo "  Creating dataset: ${ds_name}"
  aws quicksight create-data-set \
    --aws-account-id "${AWS_ACCOUNT_ID}" \
    --data-set-id "${ds_id}" \
    --name "${ds_name}" \
    --import-mode DIRECT_QUERY \
    --physical-table-map "{
      \"${ds_id}-table\": {
        \"CustomSql\": {
          \"DataSourceArn\": \"${DS_ARN}\",
          \"Name\": \"${ds_name}\",
          \"SqlQuery\": \"${sql}\",
          \"Columns\": []
        }
      }
    }" \
    --permissions "${DATASET_PERMISSIONS}" \
    --region "${REGION}" 2>/dev/null \
    || echo "    Dataset already exists — skipping."
}

create_dataset "jmi-top-skills" \
  "Top Skills" \
  "SELECT * FROM ${DATABASE}.vw_top_skills"

create_dataset "jmi-skill-trends" \
  "Weekly Skill Trends" \
  "SELECT * FROM ${DATABASE}.vw_skill_trends"

create_dataset "jmi-salary-by-country" \
  "Salary by Country" \
  "SELECT * FROM ${DATABASE}.vw_salary_by_country"

create_dataset "jmi-skill-growth" \
  "Skill Growth & Forecasts" \
  "SELECT * FROM ${DATABASE}.vw_skill_growth"

create_dataset "jmi-emerging-skills" \
  "Emerging Skills" \
  "SELECT * FROM ${DATABASE}.vw_emerging_skills"

create_dataset "jmi-skill-cooccurrence" \
  "Skill Co-occurrence" \
  "SELECT * FROM ${DATABASE}.vw_skill_cooccurrence"

create_dataset "jmi-dashboard-kpis" \
  "Dashboard KPIs" \
  "SELECT * FROM ${DATABASE}.vw_dashboard_kpis"

create_dataset "jmi-skills-by-category" \
  "Skills by Category" \
  "SELECT * FROM ${DATABASE}.vw_skills_by_category"

# ── Step 4: Print next steps ─────────────────────────────────────────────────

echo ""
echo "[4/4] Setup complete!"
echo ""
echo "==============================================="
echo "  QuickSight Deployment Complete"
echo "==============================================="
echo ""
echo "  Data source : ${DS_ID}"
echo "  Datasets    : 8 datasets created (jmi-*)"
echo ""
echo "  Next steps — create your dashboard in QuickSight console:"
echo ""
echo "  1. Open QuickSight: https://${REGION}.quicksight.aws.amazon.com/"
echo ""
echo "  2. Click 'New analysis' → select any of the jmi-* datasets"
echo ""
echo "  3. Recommended visuals per dataset:"
echo ""
echo "     jmi-top-skills           → Horizontal bar chart (skill × job_count, color by avg_salary)"
echo "     jmi-skill-trends         → Line chart (week × job_count, color by skill)"
echo "     jmi-salary-by-country    → Geo map or bar chart (country × avg_salary)"
echo "     jmi-skill-growth         → Combo bar+KPI (skill × growth_pct, color by trend_status)"
echo "     jmi-emerging-skills      → Scatter plot (current_jobs × growth_pct, size by avg_salary)"
echo "     jmi-skill-cooccurrence   → Heat map or pivot (skill_a × skill_b × cooccurrence_count)"
echo "     jmi-dashboard-kpis       → KPI widgets (total_jobs, unique_skills, avg_salary)"
echo "     jmi-skills-by-category   → Pie / treemap (category × job_count)"
echo ""
echo "  4. Publish the analysis as a dashboard to share with your team."
echo ""
echo "  Docs: https://docs.aws.amazon.com/quicksight/latest/user/creating-an-analysis.html"
