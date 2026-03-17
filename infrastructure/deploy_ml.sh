#!/bin/bash
#===============================================================================
# SageMaker ML Pipeline Deployment
#
# Deploys the skill demand forecasting ML pipeline:
#   1. SageMaker IAM role
#   2. Feature generation (Athena CTAS)
#   3. Training job (XGBoost)
#   4. Model registration
#   5. Inference endpoint (optional)
#
# Usage:
#   bash infrastructure/deploy_ml.sh                    # train only
#   bash infrastructure/deploy_ml.sh --create-endpoint  # train + endpoint
#===============================================================================
set -euo pipefail

if [[ -f .env ]]; then set -a; source .env; set +a; fi

BUCKET_NAME="${S3_BUCKET:?Set S3_BUCKET in .env or export it}"
CREATE_ENDPOINT="${1:-}"
REGION="${AWS_REGION:-us-east-1}"
ROLE_NAME="SageMakerJobMarketRole"
MODEL_NAME="skill-demand-forecaster"
ENDPOINT_NAME="skill-demand-endpoint"

echo "==============================================="
echo "  SageMaker ML Pipeline Deployment"
echo "==============================================="
echo "  Bucket : ${BUCKET_NAME}"
echo "  Region : ${REGION}"
echo ""

# ── 1. IAM Role ─────────────────────────────────────────────────────────────

echo "[1/6] Creating SageMaker IAM role …"

cat > /tmp/sagemaker-trust.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "sagemaker.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
EOF

aws iam create-role \
  --role-name "${ROLE_NAME}" \
  --assume-role-policy-document file:///tmp/sagemaker-trust.json \
  2>/dev/null || echo "  Role already exists."

for policy in AmazonSageMakerFullAccess AmazonS3FullAccess AmazonAthenaFullAccess; do
  aws iam attach-role-policy \
    --role-name "${ROLE_NAME}" \
    --policy-arn "arn:aws:iam::aws:policy/${policy}" 2>/dev/null || true
done

ROLE_ARN="$(aws iam get-role --role-name "${ROLE_NAME}" --query 'Role.Arn' --output text)"
echo "  Role ARN: ${ROLE_ARN}"
sleep 10

# ── 2. Upload training scripts ──────────────────────────────────────────────

echo "[2/6] Uploading ML scripts to S3 …"

WORK_DIR="$(mktemp -d)"
mkdir -p "${WORK_DIR}/code"

cp ml/training/train.py        "${WORK_DIR}/code/train.py"
cp ml/training/feature_engineering.py "${WORK_DIR}/code/feature_engineering.py"

cat > "${WORK_DIR}/code/requirements.txt" << 'REQ'
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
pyarrow>=14.0.0
REQ

(cd "${WORK_DIR}" && tar -czf sourcedir.tar.gz code/)
aws s3 cp "${WORK_DIR}/sourcedir.tar.gz" "s3://${BUCKET_NAME}/ml/code/sourcedir.tar.gz"
rm -rf "${WORK_DIR}"
echo "  Scripts uploaded."

# ── 3. Generate training features via Athena ─────────────────────────────────

echo "[3/6] Generating training features …"

aws athena start-query-execution \
  --query-string "DROP TABLE IF EXISTS job_market_db.ml_training_data" \
  --result-configuration "OutputLocation=s3://${BUCKET_NAME}/athena-results/" \
  --region "${REGION}" >/dev/null
sleep 5

read -r -d '' FEATURE_SQL << 'FSQL' || true
CREATE TABLE job_market_db.ml_training_data
WITH (format = 'PARQUET', external_location = 's3://BUCKET_PLACEHOLDER/ml/features/')
AS
WITH weekly_stats AS (
    SELECT
        skill,
        DATE_TRUNC('week', posted_date) AS week,
        COUNT(*)            AS job_count,
        AVG(salary_mid_usd) AS avg_salary
    FROM job_market_db.job_skills
    GROUP BY skill, DATE_TRUNC('week', posted_date)
),
features AS (
    SELECT
        skill,
        week,
        job_count,
        avg_salary,
        LAG(job_count, 1) OVER w AS job_count_lag_1w,
        LAG(job_count, 4) OVER w AS job_count_lag_4w,
        LAG(avg_salary, 1) OVER w AS salary_lag_1w,
        AVG(job_count) OVER (PARTITION BY skill ORDER BY week
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS job_count_ma_4w,
        AVG(job_count) OVER (PARTITION BY skill ORDER BY week
            ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) AS job_count_ma_8w,
        (job_count - LAG(job_count, 1) OVER w) * 1.0
            / NULLIF(LAG(job_count, 1) OVER w, 0) AS wow_growth,
        (job_count - LAG(job_count, 4) OVER w) * 1.0
            / NULLIF(LAG(job_count, 4) OVER w, 0) AS mom_growth,
        LEAD(job_count, 4) OVER w AS target
    FROM weekly_stats
    WINDOW w AS (PARTITION BY skill ORDER BY week)
)
SELECT * FROM features WHERE target IS NOT NULL
FSQL

FEATURE_SQL="${FEATURE_SQL//BUCKET_PLACEHOLDER/${BUCKET_NAME}}"

QUERY_ID="$(aws athena start-query-execution \
  --query-string "${FEATURE_SQL}" \
  --result-configuration "OutputLocation=s3://${BUCKET_NAME}/athena-results/" \
  --region "${REGION}" \
  --query 'QueryExecutionId' --output text)"

echo "  Athena query: ${QUERY_ID}"
while true; do
  STATUS="$(aws athena get-query-execution \
    --query-execution-id "${QUERY_ID}" \
    --query 'QueryExecution.Status.State' --output text \
    --region "${REGION}")"
  echo "  Status: ${STATUS}"
  [[ "${STATUS}" == "SUCCEEDED" || "${STATUS}" == "FAILED" ]] && break
  sleep 10
done

[[ "${STATUS}" == "FAILED" ]] && echo "  WARNING: feature generation may have failed."

# ── 4. SageMaker training job ────────────────────────────────────────────────

echo "[4/6] Starting SageMaker training job …"

TRAINING_JOB="skill-forecaster-$(date +%Y%m%d-%H%M%S)"

aws sagemaker create-training-job \
  --training-job-name "${TRAINING_JOB}" \
  --role-arn "${ROLE_ARN}" \
  --algorithm-specification '{
    "TrainingImage":"683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.7-1",
    "TrainingInputMode":"File"
  }' \
  --input-data-config "[{
    \"ChannelName\":\"train\",
    \"DataSource\":{\"S3DataSource\":{
      \"S3DataType\":\"S3Prefix\",
      \"S3Uri\":\"s3://${BUCKET_NAME}/ml/features/\",
      \"S3DataDistributionType\":\"FullyReplicated\"
    }},
    \"ContentType\":\"application/x-parquet\"
  }]" \
  --output-data-config "{\"S3OutputPath\":\"s3://${BUCKET_NAME}/ml/models/\"}" \
  --resource-config '{"InstanceType":"ml.m5.xlarge","InstanceCount":1,"VolumeSizeInGB":10}' \
  --stopping-condition '{"MaxRuntimeInSeconds":3600}' \
  --hyper-parameters '{"objective":"reg:squarederror","num_round":"200","max_depth":"6","eta":"0.05","subsample":"0.8"}' \
  --region "${REGION}"

echo "  Job: ${TRAINING_JOB}"
echo "  Waiting for training …"

while true; do
  STATUS="$(aws sagemaker describe-training-job \
    --training-job-name "${TRAINING_JOB}" \
    --query 'TrainingJobStatus' --output text --region "${REGION}")"
  echo "  Status: ${STATUS}"
  [[ "${STATUS}" == "Completed" || "${STATUS}" == "Failed" || "${STATUS}" == "Stopped" ]] && break
  sleep 30
done

if [[ "${STATUS}" != "Completed" ]]; then
  echo "ERROR: training ${STATUS}."
  exit 1
fi

MODEL_ARTIFACT="$(aws sagemaker describe-training-job \
  --training-job-name "${TRAINING_JOB}" \
  --query 'ModelArtifacts.S3ModelArtifacts' --output text --region "${REGION}")"
echo "  Artifact: ${MODEL_ARTIFACT}"

# ── 5. Register model ───────────────────────────────────────────────────────

echo "[5/6] Registering model …"

aws sagemaker delete-model --model-name "${MODEL_NAME}" --region "${REGION}" 2>/dev/null || true

aws sagemaker create-model \
  --model-name "${MODEL_NAME}" \
  --primary-container "{
    \"Image\":\"683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.7-1\",
    \"ModelDataUrl\":\"${MODEL_ARTIFACT}\"
  }" \
  --execution-role-arn "${ROLE_ARN}" \
  --region "${REGION}"

echo "  Model: ${MODEL_NAME}"

# ── 6. Endpoint (optional) ──────────────────────────────────────────────────

if [[ "${CREATE_ENDPOINT}" == "--create-endpoint" ]]; then
  echo "[6/6] Creating inference endpoint …"

  aws sagemaker create-endpoint-config \
    --endpoint-config-name "${MODEL_NAME}-config" \
    --production-variants "[{
      \"VariantName\":\"primary\",
      \"ModelName\":\"${MODEL_NAME}\",
      \"InstanceType\":\"ml.t2.medium\",
      \"InitialInstanceCount\":1
    }]" \
    --region "${REGION}" 2>/dev/null || echo "  Config already exists."

  aws sagemaker create-endpoint \
    --endpoint-name "${ENDPOINT_NAME}" \
    --endpoint-config-name "${MODEL_NAME}-config" \
    --region "${REGION}" 2>/dev/null || echo "  Endpoint already exists."

  echo "  Waiting for endpoint …"
  aws sagemaker wait endpoint-in-service \
    --endpoint-name "${ENDPOINT_NAME}" \
    --region "${REGION}"

  echo "  Endpoint ready: ${ENDPOINT_NAME}"
else
  echo "[6/6] Skipping endpoint (pass --create-endpoint to enable)."
fi

rm -f /tmp/sagemaker-trust.json

echo ""
echo "==============================================="
echo "  ML Pipeline Deployment Complete"
echo "==============================================="
echo "  Training Job  : ${TRAINING_JOB}"
echo "  Model         : ${MODEL_NAME}"
echo "  Model Artifact: ${MODEL_ARTIFACT}"
[[ "${CREATE_ENDPOINT}" == "--create-endpoint" ]] && echo "  Endpoint      : ${ENDPOINT_NAME}"
echo ""
echo "Run predictions locally:"
echo "  python -m ml.inference.predictor --model-path ml/models --action report"
echo ""
echo "Cost estimate:"
echo "  Training (1 hr/week)        : ~\$0.20"
echo "  Endpoint (if created, 24/7) : ~\$50/month (ml.t2.medium)"
echo "  Tip: use batch transform instead of an always-on endpoint."
