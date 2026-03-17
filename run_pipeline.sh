#!/bin/bash
# run_pipeline.sh - Execute the full data pipeline end to end
set -euo pipefail

if [[ -f .env ]]; then
  set -a; source .env; set +a
fi

BUCKET_NAME="${S3_BUCKET:?Set S3_BUCKET in .env or export it}"
REGION="${AWS_REGION:-us-east-1}"

echo "============================================"
echo "  Running Job Market Intelligence Pipeline"
echo "============================================"

# ── Step 1: Invoke Lambda scraper ──
echo ""
echo "Step 1/4: Collecting jobs from Adzuna..."
aws lambda invoke \
  --function-name job-market-scraper \
  --payload '{}' \
  --region "${REGION}" \
  output.json >/dev/null

echo "  Lambda result:"
cat output.json
echo ""

# ── Step 2: Run Glue ETL ──
echo "Step 2/4: Processing data with Glue ETL..."
RUN_ID=$(aws glue start-job-run --job-name job-market-etl --query 'JobRunId' --output text --region "${REGION}")
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

# ── Step 3: Repair Athena partitions ──
echo ""
echo "Step 3/4: Updating Athena partitions..."
for table in jobs job_skills; do
  aws athena start-query-execution \
    --query-string "MSCK REPAIR TABLE job_market_db.${table}" \
    --result-configuration "OutputLocation=s3://${BUCKET_NAME}/athena-results/" \
    --region "${REGION}" >/dev/null
done
sleep 10

# ── Step 4: Smoke-test query ──
echo "Step 4/4: Running test query..."
QUERY_ID=$(aws athena start-query-execution \
  --query-string "SELECT skill, COUNT(*) as cnt FROM job_market_db.job_skills GROUP BY skill ORDER BY cnt DESC LIMIT 10" \
  --result-configuration "OutputLocation=s3://${BUCKET_NAME}/athena-results/" \
  --region "${REGION}" \
  --query 'QueryExecutionId' --output text)

sleep 5
aws athena get-query-results --query-execution-id "${QUERY_ID}" --region "${REGION}" 2>/dev/null \
  || echo "  (query still running -- check Athena console)"

rm -f output.json

echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "============================================"
echo ""
echo "  View data  : aws s3 ls s3://${BUCKET_NAME}/processed/ --recursive"
echo "  Athena UI  : https://console.aws.amazon.com/athena"
echo "  Dashboard  : cd visualization/streamlit_app && streamlit run app.py"
