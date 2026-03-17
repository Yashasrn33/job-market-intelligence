#!/bin/bash
# deploy_lambda.sh - Package, deploy Lambda, and create daily schedule
set -euo pipefail

FUNCTION_NAME="job-market-scraper"
ROLE_NAME="JobMarketLambdaRole"
REGION="${AWS_REGION:-us-east-1}"
RUNTIME="python3.11"
TIMEOUT=300
MEMORY=512
BUCKET_NAME="${S3_BUCKET:?Set S3_BUCKET before running this script}"

echo "=== Deploying Lambda: ${FUNCTION_NAME} ==="

# --- Package ---
DEPLOY_DIR="lambda_package"
rm -rf "${DEPLOY_DIR}" deployment.zip
mkdir -p "${DEPLOY_DIR}"

echo "Installing dependencies..."
pip install requests -t "${DEPLOY_DIR}" --quiet

echo "Copying function code..."
cp ingestion/lambda_function.py "${DEPLOY_DIR}/"

echo "Creating zip..."
(cd "${DEPLOY_DIR}" && zip -r ../deployment.zip . -q)

# --- Deploy ---
ROLE_ARN="$(aws iam get-role --role-name "${ROLE_NAME}" --query 'Role.Arn' --output text)"
echo "Using role: ${ROLE_ARN}"

sleep 10  # allow IAM propagation on first deploy

if aws lambda get-function --function-name "${FUNCTION_NAME}" --region "${REGION}" >/dev/null 2>&1; then
  echo "Updating existing function..."
  aws lambda update-function-code \
    --function-name "${FUNCTION_NAME}" \
    --zip-file fileb://deployment.zip \
    --region "${REGION}" >/dev/null
else
  echo "Creating new function..."
  aws lambda create-function \
    --function-name "${FUNCTION_NAME}" \
    --runtime "${RUNTIME}" \
    --role "${ROLE_ARN}" \
    --handler lambda_function.lambda_handler \
    --zip-file fileb://deployment.zip \
    --timeout "${TIMEOUT}" \
    --memory-size "${MEMORY}" \
    --region "${REGION}" \
    --environment "Variables={S3_BUCKET=${BUCKET_NAME}}" >/dev/null
fi

# --- EventBridge daily schedule ---
echo "Setting up daily schedule (6 AM UTC)..."
RULE_NAME="daily-job-scrape"

aws events put-rule \
  --name "${RULE_NAME}" \
  --schedule-expression "cron(0 6 * * ? *)" \
  --state ENABLED \
  --region "${REGION}" >/dev/null

LAMBDA_ARN="$(aws lambda get-function \
  --function-name "${FUNCTION_NAME}" \
  --query 'Configuration.FunctionArn' \
  --output text \
  --region "${REGION}")"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"

aws lambda add-permission \
  --function-name "${FUNCTION_NAME}" \
  --statement-id eventbridge-invoke \
  --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn "arn:aws:events:${REGION}:${ACCOUNT_ID}:rule/${RULE_NAME}" \
  --region "${REGION}" 2>/dev/null || true

aws events put-targets \
  --rule "${RULE_NAME}" \
  --targets "Id=1,Arn=${LAMBDA_ARN}" \
  --region "${REGION}" >/dev/null

# --- Cleanup ---
rm -rf "${DEPLOY_DIR}" deployment.zip

echo ""
echo "=== Deployment complete ==="
echo "  Function : ${FUNCTION_NAME}"
echo "  Schedule : Daily at 6:00 AM UTC"
echo ""
echo "Test manually:"
echo "  aws lambda invoke --function-name ${FUNCTION_NAME} output.json && cat output.json"
echo ""
echo "View logs:"
echo "  aws logs tail /aws/lambda/${FUNCTION_NAME} --follow"
