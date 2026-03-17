# AI-Powered Job Market Intelligence System

End-to-end pipeline: **Adzuna API → Lambda Scraper → S3 Data Lake → Glue ETL → Athena Analytics → Streamlit Dashboard**

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
│   └── cloudformation/
│       └── stack.yaml            # Optional IaC template
│
├── ingestion/                     # Data Collection Layer
│   ├── lambda_function.py        # Adzuna scraper (Lambda)
│   ├── requirements.txt          # Lambda dependencies
│   └── scrapers/                 # Reusable scraper modules
│       ├── adzuna.py
│       ├── jsearch.py
│       └── kaggle_loader.py
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
├── visualization/                 # Streamlit Dashboard
│   ├── requirements.txt
│   └── streamlit_app/
│       ├── app.py
│       ├── pages/
│       │   ├── skill_trends.py
│       │   ├── salary_map.py
│       │   └── forecasts.py
│       └── components/
│           └── charts.py
│
└── tests/
    ├── test_ingestion.py
    ├── test_processing.py
    └── fixtures/
        └── sample_jobs.json
```

## Quick Start

### 1. Local Setup (5 min)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure AWS

```bash
aws configure          # enter your access key, secret, region=us-east-1

cp .env.example .env   # or edit .env directly
# Set S3_BUCKET and Adzuna credentials
```

### 3. Deploy Everything (one command)

```bash
bash setup.sh
```

### 4. Run the Pipeline

```bash
bash run_pipeline.sh
```

### 5. Launch Dashboard

```bash
cd visualization/streamlit_app
streamlit run app.py
# opens at http://localhost:8501
```

## Run Tests

```bash
pytest -q
```

## Cost Estimate

| Service | Monthly Est. |
|---------|-------------|
| S3 (10 GB) | $0.23 |
| Lambda (1 000 runs) | $0.20 |
| Glue (2 DPU-hr/day) | $13.20 |
| Athena (10 GB/day) | $1.50 |
| Secrets Manager | $0.40 |
| **Total** | **~$16/month** |
