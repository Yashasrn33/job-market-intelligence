#!/bin/bash
# run_pipeline.sh - Execute the full data pipeline end to end
#
# Usage:
#   bash run_pipeline.sh             # ingest + ETL + Athena only
#   bash run_pipeline.sh --with-ml   # also runs ML feature gen + training
set -euo pipefail

if [[ -f .env ]]; then
  set -a; source .env; set +a
fi

BUCKET_NAME="${S3_BUCKET:?Set S3_BUCKET in .env or export it}"
REGION="${AWS_REGION:-us-east-1}"
WITH_ML="${1:-}"

TOTAL_STEPS=5
[[ "${WITH_ML}" == "--with-ml" ]] && TOTAL_STEPS=6

echo "============================================"
echo "  Running Job Market Intelligence Pipeline"
echo "============================================"

# ── Step 1: Invoke Lambda scraper ──────────────────────────────────────────
echo ""
echo "Step 1/${TOTAL_STEPS}: Collecting jobs from Adzuna..."
EXECUTION_DATE="$(date -u +%F)"
RAW_KEY="raw/jobs/source=adzuna/date=${EXECUTION_DATE}/jobs.json"

echo "  Invoking Lambda (async) ..."
aws lambda invoke \
  --function-name job-market-scraper \
  --invocation-type Event \
  --payload '{}' \
  --region "${REGION}" \
  /dev/null >/dev/null

echo "  Waiting for raw data in S3:"
echo "    s3://${BUCKET_NAME}/${RAW_KEY}"

MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-420}"
SLEEP_SECONDS="${SLEEP_SECONDS:-10}"
ELAPSED=0

while true; do
  if aws s3api head-object --bucket "${BUCKET_NAME}" --key "${RAW_KEY}" --region "${REGION}" >/dev/null 2>&1; then
    echo "  ✓ Raw data found."
    break
  fi

  if (( ELAPSED >= MAX_WAIT_SECONDS )); then
    echo "ERROR: Timed out waiting for raw data after ${MAX_WAIT_SECONDS}s." >&2
    echo "Check Lambda logs:" >&2
    echo "  aws logs tail /aws/lambda/job-market-scraper --region ${REGION} --since 30m --follow" >&2
    exit 1
  fi

  sleep "${SLEEP_SECONDS}"
  ELAPSED=$((ELAPSED + SLEEP_SECONDS))
done

# ── Step 2: Run Glue ETL ──────────────────────────────────────────────────
echo "Step 2/${TOTAL_STEPS}: Processing data with Glue ETL..."
RUN_ID=$(aws glue start-job-run \
  --job-name job-market-etl \
  --query 'JobRunId' --output text \
  --region "${REGION}")
echo "  Job run ID: ${RUN_ID}"

echo "  Waiting for Glue job to complete..."
while true; do
  STATUS=$(aws glue get-job-run \
    --job-name job-market-etl \
    --run-id "${RUN_ID}" \
    --query 'JobRun.JobRunState' \
    --output text \
    --region "${REGION}")
  echo "  Status: ${STATUS}"
  if [[ "${STATUS}" == "SUCCEEDED" || "${STATUS}" == "FAILED" || "${STATUS}" == "STOPPED" ]]; then
    break
  fi
  sleep 30
done

if [[ "${STATUS}" != "SUCCEEDED" ]]; then
  echo "ERROR: Glue job ${STATUS}. Check the AWS console for details."
  exit 1
fi

# ── Step 3: Repair Athena partitions ───────────────────────────────────────
echo ""
echo "Step 3/${TOTAL_STEPS}: Updating Athena partitions..."
for table in jobs job_skills; do
  aws athena start-query-execution \
    --query-string "MSCK REPAIR TABLE job_market_db.${table}" \
    --result-configuration "OutputLocation=s3://${BUCKET_NAME}/athena-results/" \
    --region "${REGION}" >/dev/null
done
sleep 10

# ── Step 4: Smoke-test query ──────────────────────────────────────────────
echo "Step 4/${TOTAL_STEPS}: Running test query..."
QUERY_ID=$(aws athena start-query-execution \
  --query-string "SELECT skill, COUNT(*) as cnt FROM job_market_db.job_skills GROUP BY skill ORDER BY cnt DESC LIMIT 10" \
  --result-configuration "OutputLocation=s3://${BUCKET_NAME}/athena-results/" \
  --region "${REGION}" \
  --query 'QueryExecutionId' --output text)

sleep 5
aws athena get-query-results \
  --query-execution-id "${QUERY_ID}" \
  --region "${REGION}" 2>/dev/null \
  || echo "  (query still running -- check Athena console)"

# ── Step 5: Refresh Athena views for QuickSight ──────────────────────────
echo ""
echo "Step 5/${TOTAL_STEPS}: Refreshing QuickSight Athena views..."
bash infrastructure/deploy_quicksight.sh --views-only

# ── Step 6 (optional): ML training ────────────────────────────────────────
if [[ "${WITH_ML}" == "--with-ml" ]]; then
  echo ""
  echo "Step 6/${TOTAL_STEPS}: Running ML pipeline (feature gen + training)..."
  bash infrastructure/deploy_ml.sh
fi

echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "============================================"
echo ""
echo "  View data  : aws s3 ls s3://${BUCKET_NAME}/processed/ --recursive"
echo "  Athena UI  : https://console.aws.amazon.com/athena"
echo "  Dashboard  : bash infrastructure/deploy_quicksight.sh"
[[ "${WITH_ML}" != "--with-ml" ]] && echo "  ML training : bash run_pipeline.sh --with-ml"
