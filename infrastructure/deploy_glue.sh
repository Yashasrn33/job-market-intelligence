#!/bin/bash
# deploy_glue.sh - Upload Glue ETL script and create the Glue job
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
BUCKET_NAME="${S3_BUCKET:?Set S3_BUCKET before running this script}"

echo "=== Deploying Glue ETL job ==="

# --- Upload script ---
echo "Uploading ETL script to S3..."
aws s3 cp processing/glue_jobs/job_processor.py "s3://${BUCKET_NAME}/scripts/"

# --- Glue IAM role ---
echo "Creating Glue IAM role..."
cat > /tmp/glue-trust.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "glue.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
EOF

aws iam create-role \
  --role-name AWSGlueServiceRole-JobMarket \
  --assume-role-policy-document file:///tmp/glue-trust.json 2>/dev/null || echo "  Glue role already exists."

aws iam attach-role-policy \
  --role-name AWSGlueServiceRole-JobMarket \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole 2>/dev/null || true

cat > /tmp/glue-s3.json << POLICY
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["s3:*"],
    "Resource": ["arn:aws:s3:::${BUCKET_NAME}","arn:aws:s3:::${BUCKET_NAME}/*"]
  }]
}
POLICY

aws iam put-role-policy \
  --role-name AWSGlueServiceRole-JobMarket \
  --policy-name S3Access \
  --policy-document file:///tmp/glue-s3.json

sleep 10  # propagation

GLUE_ROLE_ARN="$(aws iam get-role --role-name AWSGlueServiceRole-JobMarket --query 'Role.Arn' --output text)"

# --- Create Glue job ---
echo "Creating Glue job: job-market-etl"
aws glue create-job \
  --name job-market-etl \
  --role "${GLUE_ROLE_ARN}" \
  --command "{
    \"Name\":\"glueetl\",
    \"ScriptLocation\":\"s3://${BUCKET_NAME}/scripts/job_processor.py\",
    \"PythonVersion\":\"3\"
  }" \
  --default-arguments "{
    \"--S3_BUCKET\":\"${BUCKET_NAME}\",
    \"--DATABASE\":\"job_market_db\",
    \"--TempDir\":\"s3://${BUCKET_NAME}/temp/\"
  }" \
  --glue-version "4.0" \
  --number-of-workers 2 \
  --worker-type "G.1X" \
  --timeout 30 2>/dev/null || echo "  Job already exists -- updating script location."

rm -f /tmp/glue-trust.json /tmp/glue-s3.json

echo ""
echo "=== Glue ETL deployment complete ==="
echo "  Job name    : job-market-etl"
echo "  Script      : s3://${BUCKET_NAME}/scripts/job_processor.py"
echo ""
echo "Run manually:"
echo "  aws glue start-job-run --job-name job-market-etl"
