#!/bin/bash
# setup_aws.sh - Create S3, IAM, Secrets Manager, and Glue Database
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
BUCKET_NAME="${S3_BUCKET:-}"
ROLE_NAME="JobMarketLambdaRole"

if [[ -z "${BUCKET_NAME}" ]]; then
  BUCKET_NAME="job-market-intelligence-$(date +%s)"
  echo "Generated bucket name: ${BUCKET_NAME}"
fi

echo "=== Setting up AWS resources in ${REGION} ==="

# --- S3 Bucket ---
echo "Creating S3 bucket: ${BUCKET_NAME}"
if aws s3api head-bucket --bucket "${BUCKET_NAME}" 2>/dev/null; then
  echo "  Bucket already exists."
else
  if [[ "${REGION}" == "us-east-1" ]]; then
    aws s3 mb "s3://${BUCKET_NAME}" --region "${REGION}"
  else
    aws s3api create-bucket --bucket "${BUCKET_NAME}" --region "${REGION}" \
      --create-bucket-configuration LocationConstraint="${REGION}"
  fi
fi

echo "Creating S3 folder structure..."
for folder in raw/jobs raw/skills processed/jobs processed/job_skills analytics/trends analytics/forecasts models scripts logs athena-results temp; do
  aws s3api put-object --bucket "${BUCKET_NAME}" --key "${folder}/" >/dev/null
done

echo "Enabling bucket versioning..."
aws s3api put-bucket-versioning \
  --bucket "${BUCKET_NAME}" \
  --versioning-configuration Status=Enabled

# --- Secrets Manager ---
SECRET_ID="job-market/adzuna-api"
echo "Creating Secrets Manager secret: ${SECRET_ID}"
if aws secretsmanager describe-secret --secret-id "${SECRET_ID}" --region "${REGION}" >/dev/null 2>&1; then
  echo "  Secret already exists."
else
  aws secretsmanager create-secret \
    --name "${SECRET_ID}" \
    --description "Adzuna API credentials" \
    --secret-string '{"app_id":"YOUR_ADZUNA_APP_ID","app_key":"YOUR_ADZUNA_APP_KEY"}' \
    --region "${REGION}"
  echo "  Created secret with placeholder values -- update before running Lambda."
fi

# --- IAM Role for Lambda ---
echo "Creating IAM role: ${ROLE_NAME}"
cat > /tmp/lambda-trust.json << 'TRUST'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "lambda.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
TRUST

if aws iam get-role --role-name "${ROLE_NAME}" >/dev/null 2>&1; then
  echo "  Role already exists."
else
  aws iam create-role \
    --role-name "${ROLE_NAME}" \
    --assume-role-policy-document file:///tmp/lambda-trust.json
fi

cat > /tmp/lambda-policy.json << POLICY
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:PutObject","s3:GetObject","s3:ListBucket"],
      "Resource": ["arn:aws:s3:::${BUCKET_NAME}","arn:aws:s3:::${BUCKET_NAME}/*"]
    },
    {
      "Effect": "Allow",
      "Action": ["secretsmanager:GetSecretValue"],
      "Resource": "arn:aws:secretsmanager:*:*:secret:job-market/*"
    },
    {
      "Effect": "Allow",
      "Action": ["logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents"],
      "Resource": "arn:aws:logs:*:*:*"
    }
  ]
}
POLICY

aws iam put-role-policy \
  --role-name "${ROLE_NAME}" \
  --policy-name JobMarketPermissions \
  --policy-document file:///tmp/lambda-policy.json

# --- Glue Database ---
echo "Creating Glue database: job_market_db"
aws glue create-database \
  --database-input '{"Name":"job_market_db","Description":"Job market intelligence"}' \
  2>/dev/null || echo "  Database already exists."

# --- Glue Catalog Tables ---
echo "Creating Glue catalog table: jobs"
aws glue create-table \
  --database-name job_market_db \
  --table-input '{
    "Name":"jobs",
    "StorageDescriptor":{
      "Columns":[
        {"Name":"job_id","Type":"string"},
        {"Name":"title","Type":"string"},
        {"Name":"company","Type":"string"},
        {"Name":"description","Type":"string"},
        {"Name":"location","Type":"string"},
        {"Name":"country","Type":"string"},
        {"Name":"salary_min_usd","Type":"double"},
        {"Name":"salary_max_usd","Type":"double"},
        {"Name":"salary_mid_usd","Type":"double"},
        {"Name":"category","Type":"string"},
        {"Name":"url","Type":"string"},
        {"Name":"posted_date","Type":"date"},
        {"Name":"extracted_skills","Type":"array<string>"},
        {"Name":"skill_count","Type":"int"}
      ],
      "Location":"s3://'"${BUCKET_NAME}"'/processed/jobs/",
      "InputFormat":"org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
      "OutputFormat":"org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat",
      "SerdeInfo":{"SerializationLibrary":"org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"}
    },
    "PartitionKeys":[{"Name":"year","Type":"int"},{"Name":"month","Type":"int"}],
    "TableType":"EXTERNAL_TABLE"
  }' 2>/dev/null || echo "  Table 'jobs' already exists."

echo "Creating Glue catalog table: job_skills"
aws glue create-table \
  --database-name job_market_db \
  --table-input '{
    "Name":"job_skills",
    "StorageDescriptor":{
      "Columns":[
        {"Name":"job_id","Type":"string"},
        {"Name":"posted_date","Type":"date"},
        {"Name":"country","Type":"string"},
        {"Name":"salary_mid_usd","Type":"double"},
        {"Name":"skill","Type":"string"}
      ],
      "Location":"s3://'"${BUCKET_NAME}"'/processed/job_skills/",
      "InputFormat":"org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
      "OutputFormat":"org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat",
      "SerdeInfo":{"SerializationLibrary":"org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"}
    },
    "PartitionKeys":[{"Name":"year","Type":"int"},{"Name":"month","Type":"int"}],
    "TableType":"EXTERNAL_TABLE"
  }' 2>/dev/null || echo "  Table 'job_skills' already exists."

rm -f /tmp/lambda-trust.json /tmp/lambda-policy.json

echo ""
echo "=== AWS setup complete ==="
echo "  S3 Bucket : ${BUCKET_NAME}"
echo "  IAM Role  : ${ROLE_NAME}"
echo "  Secret    : ${SECRET_ID}"
echo "  Glue DB   : job_market_db"
echo ""
echo "Export for subsequent scripts:"
echo "  export S3_BUCKET=${BUCKET_NAME}"
