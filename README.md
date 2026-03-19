# AI-Powered Job Market Intelligence System

End-to-end pipeline: **Adzuna API → Lambda Scraper → S3 Data Lake → Glue ETL → Athena Analytics → QuickSight Dashboard**

## Repository Structure

```
job-market-intelligence/
├── setup.sh                       # One-click AWS setup
├── run_pipeline.sh                # Execute full pipeline
├── requirements.txt               # Python dependencies
│
├── infrastructure/                # AWS Infrastructure
│   ├── setup_aws.sh              # S3, IAM, Secrets, Glue DB
│   ├── deploy_lambda.sh          # Lambda deployment
│   ├── deploy_glue.sh            # Glue ETL deployment
│   ├── deploy_quicksight.sh     # QuickSight dashboard setup
│   └── cloudformation/
│       └── stack.yaml            # Optional IaC template
│
├── ingestion/                     # Data Collection Layer
│   ├── lambda_function.py        # Adzuna scraper (Lambda)
│   ├── requirements.txt          # Lambda dependencies
│   └── scrapers/                 # Reusable scraper modules
│       ├── adzuna.py             # Lightweight Lambda scraper
│       ├── adzuna_scraper.py     # Full production scraper (standalone)
│       ├── remoteok.py           # RemoteOK scraper (no API key needed)
│       └── jsearch.py
│
├── processing/                    # ETL & Transformation
│   ├── skill_extractor.py        # NLP skill extraction (500+ skills)
│   ├── glue_jobs/
│   │   └── job_processor.py      # PySpark Glue ETL job
│   └── schemas/
│       └── job_schema.py         # Data models
│
├── analytics/                     # SQL & Analysis
│   ├── athena_queries/
│   │   ├── top_skills.sql
│   │   ├── skill_trends.sql
│   │   ├── geographic_demand.sql
│   │   └── salary_by_skill.sql
│   └── notebooks/
│       └── exploratory.ipynb
│
├── ml/                            # Machine Learning
│   ├── training/
│   │   ├── train.py              # Main training entrypoint
│   │   ├── feature_engineering.py
│   │   ├── skill_forecast.py
│   │   ├── skill_embedding.py
│   │   └── emerging_detector.py
│   └── inference/
│       └── predictor.py
│
├── visualization/                 # QuickSight Dashboard
│   └── quicksight/
│       └── athena_views.sql      # Athena views powering QuickSight datasets
│
└── tests/
    ├── test_ingestion.py
    ├── test_processing.py
    └── fixtures/
        └── sample_jobs.json
```

## Quick Start – End-to-End Pipeline

### 0. Local Setup (Python & deps)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# (macOS only, for XGBoost)
brew install libomp
```

### 1. Configure AWS & Environment

```bash
aws configure          # access key, secret, region (e.g. us-east-1)

cp .env.example .env   # create your local config
```

Then edit `.env` and set:

- `AWS_REGION` (e.g. `us-east-1`)
- `S3_BUCKET` (optional – leave blank for auto-generated name)
- `ADZUNA_APP_ID` / `ADZUNA_APP_KEY` (from https://developer.adzuna.com)
- `GLUE_DATABASE=job_market_db` (default)

### 2. One-Click AWS Setup (S3 + IAM + Secrets + Glue + Lambda + Glue Job)

```bash
bash setup.sh
```

This will:
- Create/verify the S3 bucket and basic folder structure
- Create IAM roles for Lambda, Glue, and Glue DB/tables
- Create a Secrets Manager secret `job-market/adzuna-api` with placeholder Adzuna keys
- Deploy the `job-market-scraper` Lambda and a daily EventBridge schedule (6 AM UTC)
- Deploy the `job-market-etl` Glue job

After setup, update the secret with your real Adzuna credentials:

```bash
aws secretsmanager put-secret-value \
  --secret-id job-market/adzuna-api \
  --secret-string '{"app_id":"YOUR_REAL_ID","app_key":"YOUR_REAL_KEY"}'
```

### 3. Data Ingestion Options

> **⚠️ IMPORTANT: Choose ONE ingestion method to avoid partition conflicts.**
> 
> The Lambda writes JSON with `date=YYYY-MM-DD` partitions, while the standalone scraper writes Parquet with `year=YYYY/month=M` partitions. **Do not mix them** — the Glue ETL job will fail with "Conflicting partition column names detected" if both exist.

#### Option A: Lambda Ingestion (Recommended for Production)

Best for: Daily automated ingestion, smaller batches

```bash
# Run the full pipeline (Lambda → Glue ETL → Athena)
bash run_pipeline.sh
```

#### Option B: Standalone Scraper (Recommended for Bulk Collection)

Best for: Initial data load, bulk historical scraping, local development

```bash
BUCKET=$(grep S3_BUCKET .env | cut -d= -f2)

# Clear any existing raw data first to avoid conflicts
aws s3 rm "s3://${BUCKET}/raw/jobs/source=adzuna/" --recursive

# Run bulk scraper
python -m ingestion.scrapers.adzuna_scraper \
  --bucket "${BUCKET}" \
  --countries us,gb,ca \
  --max-pages 20 \
  --max-days-old 30
```

This will:
- Query Adzuna API for tech jobs across categories and search terms
- Extract skills from job descriptions
- Write Parquet to `s3://$BUCKET/raw/jobs/source=adzuna/year=YYYY/month=M/`

Then run the pipeline without re-ingesting:

```bash
bash run_pipeline.sh --skip-ingest --with-ml
```

#### Option C: RemoteOK (No API Key Required)

Best for: Testing the pipeline, when Adzuna quota is exhausted

```bash
# Test the API first
python -m ingestion.scrapers.remoteok --dry-run

# Full ingestion to S3
BUCKET=$(grep S3_BUCKET .env | cut -d= -f2)
python -m ingestion.scrapers.remoteok --s3-bucket "${BUCKET}"
```

### 4. Run the Pipeline

```bash
# Full pipeline: ingest + ETL + Athena
bash run_pipeline.sh

# Skip ingestion (use existing raw data)
bash run_pipeline.sh --skip-ingest

# Include ML training
bash run_pipeline.sh --with-ml

# Skip ingestion + run ML
bash run_pipeline.sh --skip-ingest --with-ml
```

The pipeline script will:
1. Invoke the `job-market-scraper` Lambda (unless `--skip-ingest`)
2. Run the `job-market-etl` Glue job
3. Repair Athena partitions for `jobs` and `job_skills` tables
4. Execute a smoke-test query to confirm data is queryable
5. (Optional) Train ML models if `--with-ml` is specified

### 5. Train the ML Models

The ML pipeline trains three models:
- **XGBoost Demand Forecaster** — Predicts future job counts per skill
- **IsolationForest Emerging Detector** — Identifies skills with unusual growth
- **KMeans Skill Clusters** — Groups similar skills together

```bash
BUCKET=$(grep S3_BUCKET .env | cut -d= -f2)

python -m ml.training.train \
  --bucket "${BUCKET}" \
  --model-dir ml/models \
  --upload-s3
```

Artifacts are saved to:
- Local: `ml/models/skill_forecaster/`
- S3: `s3://$BUCKET/models/skill_forecaster/`

### 6. Run Predictions (CLI)

```bash
# Full market report
python -m ml.inference.predictor \
  --model-path ml/models \
  --action report

# Forecast specific skills
python -m ml.inference.predictor \
  --model-path ml/models \
  --action forecast \
  --skills python react aws

# Detect emerging skills
python -m ml.inference.predictor \
  --model-path ml/models \
  --action emerging

# Skill recommendations ("if you know X, learn Y")
python -m ml.inference.predictor \
  --model-path ml/models \
  --action recommend \
  --skills python sql
```

### 7. Deploy QuickSight Dashboard

**Prerequisite**: QuickSight must be activated in your AWS account (Standard or Enterprise edition).
Sign up at: https://quicksight.aws.amazon.com/

```bash
# Create Athena views + QuickSight data source + datasets
bash infrastructure/deploy_quicksight.sh

# Or create Athena views only (if you want to query them directly)
bash infrastructure/deploy_quicksight.sh --views-only
```

This creates:
- **8 Athena views** that pre-compute dashboard analytics
- **QuickSight data source** connected to Athena
- **8 QuickSight datasets** ready to build visuals from

| Dataset | Recommended Visual |
|---------|-------------------|
| `jmi-top-skills` | Horizontal bar (skill x job_count, color by avg_salary) |
| `jmi-skill-trends` | Line chart (week x job_count, color by skill) |
| `jmi-salary-by-country` | Geo map or bar (country x avg_salary) |
| `jmi-skill-growth` | Bar + KPI (skill x growth_pct, color by trend_status) |
| `jmi-emerging-skills` | Scatter (current_jobs x growth_pct, size by avg_salary) |
| `jmi-skill-cooccurrence` | Heat map (skill_a x skill_b x cooccurrence_count) |
| `jmi-dashboard-kpis` | KPI widgets (total_jobs, unique_skills, avg_salary) |
| `jmi-skills-by-category` | Pie or treemap (category x job_count) |

### 8. Run Tests

```bash
pytest -q          # all tests
```

---

## Troubleshooting

### Glue ETL Fails: "Conflicting partition column names detected"

**Cause**: Mixed partition schemes in `raw/jobs/` — both `date=` (Lambda) and `year=/month=` (standalone scraper) exist.

**Fix**:
```bash
# Check what's in raw/jobs
aws s3 ls s3://${S3_BUCKET}/raw/jobs/ --recursive --human-readable

# Remove ALL raw data and re-ingest with ONE method
aws s3 rm "s3://${S3_BUCKET}/raw/jobs/source=adzuna/" --recursive

# Then run your chosen ingestion method
```

### Glue ETL Fails: "Can't extract value from location: need struct type but got string"

**Cause**: Schema mismatch — the Glue script expected nested JSON but found flat Parquet columns.

**Fix**: Ensure `processing/glue_jobs/job_processor.py` handles both formats (the current version does). Re-upload to S3:
```bash
aws s3 cp processing/glue_jobs/job_processor.py \
  s3://${S3_BUCKET}/scripts/job_processor.py
```

### Adzuna API: "Usage limits exceeded" / AUTH_FAIL

**Cause**: Free tier quota exhausted (typically resets every 24 hours).

**Fix Options**:
1. Wait for quota reset
2. Use RemoteOK scraper (no API key needed): `python -m ingestion.scrapers.remoteok --dry-run`
3. Use existing data: `bash run_pipeline.sh --skip-ingest`

### ML Training: "No data found for source 'adzuna'"

**Cause**: The `processed/job_skills/` folder is empty — ETL hasn't run successfully.

**Fix**:
```bash
# Check if processed data exists
aws s3 ls s3://${S3_BUCKET}/processed/job_skills/ --recursive

# If empty, run the ETL first
bash run_pipeline.sh --skip-ingest

# Then run ML training
python -m ml.training.train --bucket "${S3_BUCKET}" --model-dir ml/models --upload-s3
```

### Lambda Times Out or Returns Empty Data

**Check Lambda logs**:
```bash
aws logs tail /aws/lambda/job-market-scraper --since 30m --follow
```

**Verify Secrets Manager has correct credentials**:
```bash
aws secretsmanager get-secret-value --secret-id job-market/adzuna-api \
  --query 'SecretString' --output text | jq .
```

### How to Verify Pipeline Health

```bash
# 1. Check raw data exists and has content
aws s3 ls s3://${S3_BUCKET}/raw/jobs/ --recursive --human-readable

# 2. Check processed data exists
aws s3 ls s3://${S3_BUCKET}/processed/ --recursive --human-readable

# 3. Test Athena query
aws athena start-query-execution \
  --query-string "SELECT COUNT(*) FROM job_market_db.job_skills" \
  --result-configuration "OutputLocation=s3://${S3_BUCKET}/athena-results/"

# 4. Check ML models exist
aws s3 ls s3://${S3_BUCKET}/models/skill_forecaster/
```

---

## Cost Estimate

| Service | Monthly Est. |
|---------|-------------|
| S3 (10 GB) | $0.23 |
| Lambda (1,000 runs) | $0.20 |
| Glue (2 DPU-hr/day) | $13.20 |
| Athena (10 GB/day) | $1.50 |
| Secrets Manager | $0.40 |
| QuickSight (1 author) | $24.00 |
| **Total** | **~$40/month** |

QuickSight reader sessions are $0.30/session (pay-per-use) for shared dashboards.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA INGESTION                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────────────────┐   │
│   │  Adzuna API  │     │  RemoteOK    │     │  Other Sources           │   │
│   │  (Rate Ltd)  │     │  (Free)      │     │  (JSearch, Kaggle)       │   │
│   └──────┬───────┘     └──────┬───────┘     └────────────┬─────────────┘   │
│          │                    │                          │                  │
│          ▼                    ▼                          ▼                  │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │                    Lambda / CLI Scrapers                          │     │
│   │                    (Transform → NDJSON/Parquet)                   │     │
│   └──────────────────────────────┬───────────────────────────────────┘     │
│                                  │                                          │
└──────────────────────────────────┼──────────────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              S3 DATA LAKE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   raw/jobs/source=adzuna/...     →  Glue ETL  →   processed/jobs/          │
│                                                    processed/job_skills/    │
│                                                                             │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
┌───────────────────────────────────┐   ┌───────────────────────────────────┐
│           ANALYTICS               │   │         MACHINE LEARNING          │
├───────────────────────────────────┤   ├───────────────────────────────────┤
│                                   │   │                                   │
│   Athena (SQL Queries)            │   │   Feature Engineering             │
│         │                         │   │         │                         │
│         ▼                         │   │         ▼                         │
│   QuickSight Dashboard            │   │   XGBoost / IsolationForest       │
│   - Skill Trends                  │   │   - Demand Forecasting            │
│   - Salary Analysis               │   │   - Emerging Skill Detection      │
│   - Geographic Demand             │   │   - Skill Clustering              │
│                                   │   │                                   │
└───────────────────────────────────┘   └───────────────────────────────────┘
```

---

## License

MIT