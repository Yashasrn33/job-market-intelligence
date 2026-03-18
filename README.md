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

### 3. (Optional) Bulk-Scrape Adzuna Data

Use the standalone Adzuna scraper to collect jobs across multiple countries in one batch:

```bash
BUCKET=$(grep S3_BUCKET .env | cut -d= -f2)
python -m ingestion.scrapers.adzuna_scraper \
  --bucket "${BUCKET}" \
  --countries us,gb,ca \
  --max-pages 20 \
  --max-days-old 30
```

This will:
- Query Adzuna API for tech jobs across categories and search terms
- Extract skills from job descriptions
- Write Parquet to:
  - `s3://$BUCKET/raw/jobs/source=adzuna/`
  - `s3://$BUCKET/processed/jobs/`
  - `s3://$BUCKET/processed/job_skills/`
  - `s3://$BUCKET/ml/training_data/`

To save locally instead of uploading to S3:

```bash
python -m ingestion.scrapers.adzuna_scraper --no-upload
```

### 4. Run the Live Ingestion + ETL Pipeline

```bash
bash run_pipeline.sh
```

This script:
- Invokes the `job-market-scraper` Lambda (Adzuna → raw S3)
- Runs the `job-market-etl` Glue job (raw → `processed/jobs` + `processed/job_skills`)
- Repairs partitions for `jobs` and `job_skills`
- Executes a smoke-test Athena query to confirm data is queryable

You can re-run this anytime to ingest more live data; the Lambda also runs daily via EventBridge.

### 5. Train the ML Models (Demand Forecast + Emerging Skills + Clusters)

The main training entrypoint is `ml/training/train.py`. You can call it directly or via the helper script.

**Option A – Direct call (recommended starting point)**

```bash
BUCKET=$(grep S3_BUCKET .env | cut -d= -f2)

python -m ml.training.train \
  --bucket "${BUCKET}" \
  --model-dir ml/models \
  --upload-s3
```

This will:
- Load job_skills data from S3 (Adzuna)
- Engineer time-series features (lags, rolling stats, growth, emergence score, etc.)
- Train:
  - XGBoost demand forecaster
  - IsolationForest emerging skill detector
  - KMeans skill clusters
- Save artifacts to `ml/models/` and upload to `s3://$BUCKET/models/skill_forecaster/`

**Option B – Full pipeline with ML in one command**

```bash
bash run_pipeline.sh --with-ml
```

This runs Steps 4 and 5 together (ingestion + ETL + training).

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

# Skill recommendations (\"if you know X, learn Y\")
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

Then open the QuickSight console and create an analysis:

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

QuickSight datasets use DIRECT_QUERY mode, so dashboards always reflect the latest data.

### 8. Run Tests

```bash
pytest -q          # all tests
```

Key test modules:
- `tests/test_ingestion.py` – Adzuna transform
- `tests/test_processing.py` – skill extraction
- `tests/test_ml.py` – feature engineering, training, inference, Adzuna scraper

## Cost Estimate

| Service | Monthly Est. |
|---------|-------------|
| S3 (10 GB) | $0.23 |
| Lambda (1 000 runs) | $0.20 |
| Glue (2 DPU-hr/day) | $13.20 |
| Athena (10 GB/day) | $1.50 |
| Secrets Manager | $0.40 |
| QuickSight (1 author) | $24.00 |
| **Total** | **~$40/month** |

QuickSight reader sessions are $0.30/session (pay-per-use) for shared dashboards.
