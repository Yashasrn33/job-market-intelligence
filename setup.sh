#!/bin/bash
# One-click AWS setup: infrastructure + Lambda + Glue
set -euo pipefail

_source_env() {
  if [[ -f .env ]]; then set -a; source .env; set +a; fi
}

echo "============================================"
echo "  Job Market Intelligence - AWS Setup"
echo "============================================"

_source_env

echo ""
echo "Step 1/3: AWS infrastructure (S3, IAM, Secrets, Glue DB, tables)"
bash infrastructure/setup_aws.sh

# Re-source .env because setup_aws.sh may have generated a new bucket name
_source_env

echo ""
echo "Step 2/3: Deploy Lambda scraper"
bash infrastructure/deploy_lambda.sh

echo ""
echo "Step 3/3: Deploy Glue ETL job"
bash infrastructure/deploy_glue.sh

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Update Secrets Manager with your Adzuna credentials:"
echo "     aws secretsmanager put-secret-value --secret-id job-market/adzuna-api \\"
echo "       --secret-string '{\"app_id\":\"YOUR_ID\",\"app_key\":\"YOUR_KEY\"}'"
echo ""
echo "  2. Run the pipeline:"
echo "     bash run_pipeline.sh"
echo ""
echo "  3. Deploy ML models (optional):"
echo "     bash infrastructure/deploy_ml.sh"
echo ""
echo "  4. Deploy QuickSight dashboard:"
echo "     bash infrastructure/deploy_quicksight.sh"
