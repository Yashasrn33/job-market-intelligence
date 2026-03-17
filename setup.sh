#!/bin/bash
# One-click AWS setup: infrastructure + Lambda + Glue
set -euo pipefail

echo "============================================"
echo "  Job Market Intelligence - AWS Setup"
echo "============================================"

# Source .env if present
if [[ -f .env ]]; then
  set -a; source .env; set +a
fi

echo ""
echo "Step 1/3: AWS infrastructure (S3, IAM, Secrets, Glue DB, tables)"
bash infrastructure/setup_aws.sh

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
echo "  3. Launch the dashboard:"
echo "     cd visualization/streamlit_app && streamlit run app.py"
