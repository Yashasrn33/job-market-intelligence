#!/bin/bash
#===============================================================================
# ML Pipeline Deployment
#
# Trains the skill demand forecasting models using the real training pipeline
# (ml/training/train.py) and optionally deploys a SageMaker endpoint.
#
# Modes:
#   Local training   — runs train.py directly on this machine
#   SageMaker        — launches a managed training job (--sagemaker flag)
#
# Usage:
#   bash infrastructure/deploy_ml.sh                          # local train
#   bash infrastructure/deploy_ml.sh --data-source adzuna      # explicit adzuna
#   bash infrastructure/deploy_ml.sh --sagemaker              # SM training job
#   bash infrastructure/deploy_ml.sh --create-endpoint        # SM + endpoint
#===============================================================================
set -euo pipefail

if [[ -f .env ]]; then set -a; source .env; set +a; fi

BUCKET_NAME="${S3_BUCKET:?Set S3_BUCKET in .env or export it}"
REGION="${AWS_REGION:-us-east-1}"
DATABASE="${GLUE_DATABASE:-job_market_db}"
DATA_SOURCE="adzuna"
MODEL_DIR="ml/models"
ROLE_NAME="SageMakerJobMarketRole"
MODEL_NAME="skill-demand-forecaster"
ENDPOINT_NAME="skill-demand-endpoint"
USE_SAGEMAKER=false
CREATE_ENDPOINT=false

for arg in "$@"; do
  case "${arg}" in
    --data-source=*) DATA_SOURCE="${arg#*=}" ;;
    --data-source)   shift; DATA_SOURCE="${1:-combined}" ;;
    --sagemaker)     USE_SAGEMAKER=true ;;
    --create-endpoint) USE_SAGEMAKER=true; CREATE_ENDPOINT=true ;;
    adzuna) DATA_SOURCE="${arg}" ;;
  esac
done

echo "==============================================="
echo "  ML Pipeline Deployment"
echo "==============================================="
echo "  Bucket      : ${BUCKET_NAME}"
echo "  Region      : ${REGION}"
echo "  Data source : ${DATA_SOURCE}"
echo "  Mode        : $(${USE_SAGEMAKER} && echo 'SageMaker' || echo 'Local')"
echo ""

# ── Local training ──────────────────────────────────────────────────────────

if ! ${USE_SAGEMAKER}; then
  echo "[1/3] Training models locally …"

  python -m ml.training.train \
    --bucket "${BUCKET_NAME}" \
    --data-source "${DATA_SOURCE}" \
    --database "${DATABASE}" \
    --region "${REGION}" \
    --model-dir "${MODEL_DIR}" \
    --upload-s3

  echo ""
  echo "[2/3] Verifying model artifacts …"
  for artifact in demand_model.pkl emergence_model.pkl cluster_model.pkl scaler.pkl feature_cols.json metrics.json; do
    if [[ -f "${MODEL_DIR}/${artifact}" ]]; then
      echo "  ✓ ${artifact}"
    else
      echo "  ✗ ${artifact} MISSING" >&2
    fi
  done

  echo ""
  echo "[3/3] Model artifacts uploaded to S3"
  echo "  s3://${BUCKET_NAME}/models/skill_forecaster/"

  echo ""
  echo "==============================================="
  echo "  Local Training Complete"
  echo "==============================================="
  echo ""
  echo "Run predictions:"
  echo "  python -m ml.inference.predictor --model-path ${MODEL_DIR} --action report"
  echo ""
  echo "Deploy QuickSight dashboard:"
  echo "  bash infrastructure/deploy_quicksight.sh"
  exit 0
fi

# ── SageMaker training ──────────────────────────────────────────────────────

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

echo "[2/6] Uploading ML scripts to S3 …"

WORK_DIR="$(mktemp -d)"
mkdir -p "${WORK_DIR}/code/ml/training"

cp ml/training/train.py              "${WORK_DIR}/code/ml/training/train.py"
cp ml/training/feature_engineering.py "${WORK_DIR}/code/ml/training/feature_engineering.py"
touch "${WORK_DIR}/code/ml/__init__.py"
touch "${WORK_DIR}/code/ml/training/__init__.py"

cat > "${WORK_DIR}/code/requirements.txt" << 'REQ'
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
pyarrow>=14.0.0
awswrangler>=3.4.0
REQ

(cd "${WORK_DIR}" && tar -czf sourcedir.tar.gz code/)
aws s3 cp "${WORK_DIR}/sourcedir.tar.gz" "s3://${BUCKET_NAME}/ml/code/sourcedir.tar.gz"
rm -rf "${WORK_DIR}"
echo "  Scripts uploaded."

echo "[3/6] Generating training features via train.py …"
echo "  (Running feature engineering from S3 data …)"

python -m ml.training.feature_engineering \
  --bucket "${BUCKET_NAME}" \
  --database "${DATABASE}" \
  --region "${REGION}" 2>&1 | tail -5

FEATURES_PATH="s3://${BUCKET_NAME}/ml/features/training_data.parquet"
echo "  Features at: ${FEATURES_PATH}"

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

if ${CREATE_ENDPOINT}; then
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
${CREATE_ENDPOINT} && echo "  Endpoint      : ${ENDPOINT_NAME}"
echo ""
echo "Run predictions locally:"
echo "  python -m ml.inference.predictor --model-path ml/models --action report"
echo ""
echo "Cost estimate:"
echo "  Training (1 hr/week)        : ~\$0.20"
echo "  Endpoint (if created, 24/7) : ~\$50/month (ml.t2.medium)"
echo "  Tip: use batch transform instead of an always-on endpoint."
