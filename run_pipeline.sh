#!/bin/bash
# run_pipeline.sh - Execute the full data pipeline end to end
#
# Usage:
#   bash run_pipeline.sh                  # ingest + ETL + Athena only
#   bash run_pipeline.sh --with-ml        # also runs ML feature gen + training
#   bash run_pipeline.sh --skip-ingest    # skip Lambda, use existing raw data
#   bash run_pipeline.sh --skip-ingest --with-ml
set -euo pipefail

if [[ -f .env ]]; then
  set -a; source .env; set +a
fi

BUCKET_NAME="${S3_BUCKET:?Set S3_BUCKET in .env or export it}"
REGION="${AWS_REGION:-us-east-1}"

# Parse arguments
SKIP_INGEST=false
WITH_ML=false
for arg in "$@"; do
  case $arg in
    --skip-ingest) SKIP_INGEST=true ;;
    --with-ml) WITH_ML=true ;;
  esac
done

TOTAL_STEPS=5
[[ "${WITH_ML}" == true ]] && TOTAL_STEPS=6
[[ "${SKIP_INGEST}" == true ]] && TOTAL_STEPS=$((TOTAL_STEPS - 1))

echo "============================================"
echo "  Running Job Market Intelligence Pipeline"
echo "============================================"
[[ "${SKIP_INGEST}" == true ]] && echo "  (Skipping ingestion, using existing data)"
echo ""

STEP=1

# ── Step 1: Invoke Lambda scraper (or skip) ────────────────────────────────
if [[ "${SKIP_INGEST}" == false ]]; then
  echo "Step ${STEP}/${TOTAL_STEPS}: Collecting jobs from Adzuna..."
  
  # Clean up any conflicting partition schemes first
  echo "  Cleaning up old data to avoid partition conflicts..."
  aws s3 rm "s3://${BUCKET_NAME}/raw/jobs/source=adzuna/" --recursive --quiet 2>/dev/null || true
  
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
    # Check for non-empty file
    SIZE=$(aws s3api head-object --bucket "${BUCKET_NAME}" --key "${RAW_KEY}" --region "${REGION}" \
      --query 'ContentLength' --output text 2>/dev/null || echo "0")
    
    if [[ "${SIZE}" != "0" ]] && [[ "${SIZE}" != "" ]]; then
      echo "  ✓ Raw data found (${SIZE} bytes)."
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
  
  STEP=$((STEP + 1))
else
  echo "Step -: Skipping ingestion (--skip-ingest flag set)"
  echo "  Using existing data in s3://${BUCKET_NAME}/raw/jobs/"
  
  # Verify some raw data exists
  RAW_COUNT=$(aws s3 ls "s3://${BUCKET_NAME}/raw/jobs/" --recursive | grep -c -E '\.(json|parquet)$' || echo "0")
  if [[ "${RAW_COUNT}" == "0" ]]; then
    echo "ERROR: No raw data found. Run without --skip-ingest first." >&2
    exit 1
  fi
  echo "  ✓ Found ${RAW_COUNT} raw data file(s)."
fi

# ── Step 2: Run Glue ETL ──────────────────────────────────────────────────
echo ""
echo "Step ${STEP}/${TOTAL_STEPS}: Processing data with Glue ETL..."
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
  echo "ERROR: Glue job ${STATUS}. Check the AWS console for details." >&2
  # Show recent error logs
  echo "  Recent error logs:" >&2
  aws logs filter-log-events \
    --log-group-name "/aws-glue/jobs/error" \
    --limit 5 \
    --query 'events[*].message' \
    --output text 2>/dev/null | tail -10 || true
  exit 1
fi

STEP=$((STEP + 1))

# ── Step 3: Repair Athena partitions ───────────────────────────────────────
echo ""
echo "Step ${STEP}/${TOTAL_STEPS}: Updating Athena partitions..."
for table in jobs job_skills; do
  aws athena start-query-execution \
    --query-string "MSCK REPAIR TABLE job_market_db.${table}" \
    --result-configuration "OutputLocation=s3://${BUCKET_NAME}/athena-results/" \
    --region "${REGION}" >/dev/null
done
sleep 10
STEP=$((STEP + 1))

# ── Step 4: Smoke-test query ──────────────────────────────────────────────
echo ""
echo "Step ${STEP}/${TOTAL_STEPS}: Running test query..."
QUERY_ID=$(aws athena start-query-execution \
  --query-string "SELECT skill, COUNT(*) as cnt FROM job_market_db.job_skills GROUP BY skill ORDER BY cnt DESC LIMIT 10" \
  --result-configuration "OutputLocation=s3://${BUCKET_NAME}/athena-results/" \
  --region "${REGION}" \
  --query 'QueryExecutionId' --output text)

sleep 5
echo "  Top skills:"
aws athena get-query-results \
  --query-execution-id "${QUERY_ID}" \
  --region "${REGION}" \
  --query 'ResultSet.Rows[1:6]' \
  --output table 2>/dev/null \
  || echo "  (query still running -- check Athena console)"
STEP=$((STEP + 1))

# ── Step 5: Refresh Athena views for QuickSight ──────────────────────────
echo ""
echo "Step ${STEP}/${TOTAL_STEPS}: Refreshing QuickSight Athena views..."
if [[ -f infrastructure/deploy_quicksight.sh ]]; then
  bash infrastructure/deploy_quicksight.sh --views-only 2>/dev/null || echo "  (QuickSight views skipped)"
else
  echo "  (QuickSight script not found, skipping)"
fi
STEP=$((STEP + 1))

# ── Step 6 (optional): ML training ────────────────────────────────────────
if [[ "${WITH_ML}" == true ]]; then
  echo ""
  echo "Step ${STEP}/${TOTAL_STEPS}: Running ML pipeline (feature gen + training)..."
  if [[ -f infrastructure/deploy_ml.sh ]]; then
    bash infrastructure/deploy_ml.sh
  else
    # Direct ML training
    python -m ml.training.train \
      --bucket "${BUCKET_NAME}" \
      --model-dir ml/models \
      --upload-s3
  fi
fi

echo ""
echo "============================================"
echo "  ✅ Pipeline complete!"
echo "============================================"
echo ""
echo "  View data  : aws s3 ls s3://${BUCKET_NAME}/processed/ --recursive"
echo "  Athena UI  : https://console.aws.amazon.com/athena"
echo "  Dashboard  : bash infrastructure/deploy_quicksight.sh"
[[ "${WITH_ML}" == false ]] && echo "  ML training : bash run_pipeline.sh --with-ml"
echo ""