"""
Kaggle LinkedIn Jobs Data Loader

Downloads, processes, and uploads Kaggle datasets to S3 for ML training.

Datasets used:
- asaniczka/1-3m-linkedin-jobs-and-skills-2024 (primary)
- arshkon/linkedin-job-postings (backup)

Usage:
    python -m ingestion.scrapers.kaggle_loader --bucket your-bucket-name --dataset primary

Prerequisites:
    pip install kaggle pandas pyarrow boto3 awswrangler
    # Set up ~/.kaggle/kaggle.json with your API credentials
"""

import os
import re
import shutil
import tempfile
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import boto3
import numpy as np
import pandas as pd
import awswrangler as wr

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


KAGGLE_DATASETS = {
    "primary": {
        "name": "asaniczka/1-3m-linkedin-jobs-and-skills-2024",
        "files": ["linkedin_job_postings.csv", "job_skills.csv"],
        "description": "1.3M LinkedIn jobs with pre-extracted skills",
    },
    "secondary": {
        "name": "arshkon/linkedin-job-postings",
        "files": ["postings.csv", "skills/*"],
        "description": "LinkedIn job postings 2023-2024",
    },
    "data_science": {
        "name": "asaniczka/data-science-job-postings-and-skills",
        "files": ["job_postings.csv", "job_skills.csv"],
        "description": "Data science focused jobs with skills",
    },
}

TECH_SKILLS = {
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "ruby", "php", "scala", "kotlin", "swift", "r", "sql",
    "react", "angular", "vue", "nodejs", "django", "flask", "fastapi",
    "spring", "express", "rails", "nextjs",
    "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "cassandra",
    "dynamodb", "snowflake", "bigquery", "redshift",
    "aws", "azure", "gcp", "kubernetes", "docker", "terraform", "serverless",
    "spark", "airflow", "kafka", "dbt", "hadoop", "flink",
    "machine learning", "deep learning", "tensorflow", "pytorch", "scikit-learn",
    "nlp", "computer vision", "llm", "mlops", "langchain",
    "ci/cd", "jenkins", "github actions", "gitlab", "ansible",
    "git", "linux", "agile", "scrum",
}


class KaggleDataLoader:
    """Downloads Kaggle datasets and loads them into S3 for ML training."""

    def __init__(self, bucket: str, region: str = "us-east-1"):
        self.bucket = bucket
        self.region = region
        self.s3 = boto3.client("s3", region_name=region)
        self._verify_kaggle_auth()

    def _verify_kaggle_auth(self):
        """Verify Kaggle API credentials are configured."""
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"

        if not kaggle_json.exists():
            logger.error("Kaggle credentials not found!")
            logger.error("1. Go to https://www.kaggle.com/account")
            logger.error("2. Click 'Create New API Token'")
            logger.error("3. Save kaggle.json to ~/.kaggle/kaggle.json")
            logger.error("4. Run: chmod 600 ~/.kaggle/kaggle.json")
            raise FileNotFoundError("Kaggle credentials not configured")

        os.chmod(kaggle_json, 0o600)
        logger.info("Kaggle credentials verified")

    def download_dataset(self, dataset_key: str = "primary") -> Path:
        """
        Download Kaggle dataset to temp directory.

        Args:
            dataset_key: Key from KAGGLE_DATASETS dict

        Returns:
            Path to extracted dataset directory
        """
        import kaggle  # deferred so missing package fails clearly

        if dataset_key not in KAGGLE_DATASETS:
            raise ValueError(
                f"Unknown dataset: {dataset_key}. Options: {list(KAGGLE_DATASETS.keys())}"
            )

        config = KAGGLE_DATASETS[dataset_key]
        dataset_name = config["name"]

        logger.info("Downloading: %s", dataset_name)
        logger.info("Description: %s", config["description"])

        temp_dir = Path(tempfile.mkdtemp())
        kaggle.api.dataset_download_files(dataset_name, path=temp_dir, unzip=True)

        files = list(temp_dir.rglob("*.csv"))
        logger.info("Downloaded to %s — files: %s", temp_dir, [f.name for f in files])

        return temp_dir

    @staticmethod
    def extract_skills_from_text(text: str) -> List[str]:
        """Extract skills from job description using regex patterns."""
        if pd.isna(text) or not text:
            return []

        text_lower = text.lower()
        found: set[str] = set()

        for skill in TECH_SKILLS:
            pattern = r"\b" + re.escape(skill) + r"\b"
            if re.search(pattern, text_lower):
                found.add(skill)

        return sorted(found)

    def process_linkedin_jobs(self, data_dir: Path) -> pd.DataFrame:
        """
        Process the primary LinkedIn jobs dataset.
        Standardizes schema to match the Glue ETL output.
        """
        logger.info("Processing LinkedIn jobs dataset...")

        jobs_file: Optional[Path] = None
        for pattern in ["linkedin_job_postings.csv", "postings.csv", "job_postings.csv"]:
            matches = list(data_dir.rglob(pattern))
            if matches:
                jobs_file = matches[0]
                break

        if not jobs_file:
            raise FileNotFoundError(f"No jobs CSV found in {data_dir}")

        logger.info("Loading: %s", jobs_file)

        chunks = []
        for chunk in pd.read_csv(jobs_file, chunksize=100_000, low_memory=False):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)

        logger.info("Loaded %s job postings", f"{len(df):,}")

        column_mapping = {
            "job_id": "job_id",
            "title": "title",
            "company_name": "company",
            "company": "company",
            "description": "description",
            "job_description": "description",
            "location": "location",
            "job_location": "location",
            "salary": "salary_text",
            "max_salary": "salary_max",
            "min_salary": "salary_min",
            "med_salary": "salary_mid",
            "formatted_work_type": "work_type",
            "applies": "application_count",
            "original_listed_time": "posted_date",
            "listed_time": "posted_date",
            "posting_domain": "source_domain",
        }
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

        if "job_id" not in df.columns:
            df["job_id"] = (
                df.apply(
                    lambda x: hash(
                        f"{x.get('title', '')}-{x.get('company', '')}-{x.get('location', '')}"
                    )
                    % (10**12),
                    axis=1,
                )
                .astype(str)
            )

        if "posted_date" in df.columns:
            df["posted_date"] = pd.to_datetime(df["posted_date"], errors="coerce")
        else:
            rng = np.random.default_rng(42)
            df["posted_date"] = pd.to_datetime("2024-01-01") - pd.to_timedelta(
                rng.integers(0, 365, len(df)), unit="D"
            )

        logger.info("Extracting skills from descriptions...")
        if "description" in df.columns:
            df["extracted_skills"] = df["description"].apply(self.extract_skills_from_text)
            df["skill_count"] = df["extracted_skills"].apply(len)
        else:
            df["extracted_skills"] = [[] for _ in range(len(df))]
            df["skill_count"] = 0

        for col in ["salary_min", "salary_max", "salary_mid"]:
            if col not in df.columns:
                df[col] = None
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if df["salary_mid"].isna().all() and not df["salary_min"].isna().all():
            df["salary_mid"] = (df["salary_min"].fillna(0) + df["salary_max"].fillna(0)) / 2
            df.loc[df["salary_mid"] == 0, "salary_mid"] = None

        df["source"] = "kaggle_linkedin"
        df["ingested_at"] = datetime.utcnow().isoformat()
        df["year"] = df["posted_date"].dt.year
        df["month"] = df["posted_date"].dt.month

        final_columns = [
            "job_id", "title", "company", "description", "location",
            "salary_min", "salary_max", "salary_mid",
            "posted_date", "extracted_skills", "skill_count",
            "source", "ingested_at", "year", "month",
        ]
        df = df[[c for c in final_columns if c in df.columns]]
        df = df.dropna(subset=["title"])

        logger.info(
            "Processed %s jobs with %s skill extractions",
            f"{len(df):,}",
            f"{df['skill_count'].sum():,}",
        )
        return df

    @staticmethod
    def create_job_skills_table(jobs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Explode the extracted_skills array into a job_skills junction table
        matching the Glue ETL output schema.
        """
        logger.info("Creating job_skills junction table...")

        job_skills = jobs_df[
            ["job_id", "posted_date", "salary_mid", "year", "month", "extracted_skills"]
        ].copy()
        job_skills = job_skills.explode("extracted_skills")
        job_skills = job_skills.rename(columns={"extracted_skills": "skill"})
        job_skills = job_skills.dropna(subset=["skill"])
        job_skills["country"] = "US"

        logger.info("Created %s job-skill records", f"{len(job_skills):,}")
        return job_skills

    def upload_to_s3(
        self, jobs_df: pd.DataFrame, job_skills_df: pd.DataFrame
    ) -> Dict[str, object]:
        """Upload processed data to S3 in Parquet format."""
        logger.info("Uploading to S3 bucket: %s", self.bucket)

        raw_path = f"s3://{self.bucket}/raw/jobs/source=kaggle/"
        logger.info("Writing raw jobs to: %s", raw_path)
        wr.s3.to_parquet(
            df=jobs_df,
            path=raw_path,
            dataset=True,
            partition_cols=["year", "month"],
            mode="overwrite",
        )

        processed_jobs_path = f"s3://{self.bucket}/processed/jobs_kaggle/"
        logger.info("Writing processed jobs to: %s", processed_jobs_path)
        wr.s3.to_parquet(
            df=jobs_df,
            path=processed_jobs_path,
            dataset=True,
            partition_cols=["year", "month"],
            mode="overwrite",
        )

        job_skills_path = f"s3://{self.bucket}/processed/job_skills_kaggle/"
        logger.info("Writing job_skills to: %s", job_skills_path)
        wr.s3.to_parquet(
            df=job_skills_df,
            path=job_skills_path,
            dataset=True,
            partition_cols=["year", "month"],
            mode="overwrite",
        )

        ml_features_path = f"s3://{self.bucket}/ml/training_data/"
        logger.info("Writing ML training data to: %s", ml_features_path)
        wr.s3.to_parquet(
            df=job_skills_df,
            path=ml_features_path,
            dataset=True,
            mode="overwrite",
        )

        logger.info("All data uploaded to S3")

        return {
            "raw_path": raw_path,
            "processed_jobs_path": processed_jobs_path,
            "job_skills_path": job_skills_path,
            "ml_features_path": ml_features_path,
            "jobs_count": len(jobs_df),
            "job_skills_count": len(job_skills_df),
        }

    def register_in_glue_catalog(self):
        """Register the Kaggle tables in Glue Catalog for Athena queries."""
        glue = boto3.client("glue", region_name=self.region)
        database = "job_market_db"

        jobs_columns = [
            {"Name": "job_id", "Type": "string"},
            {"Name": "title", "Type": "string"},
            {"Name": "company", "Type": "string"},
            {"Name": "description", "Type": "string"},
            {"Name": "location", "Type": "string"},
            {"Name": "salary_min", "Type": "double"},
            {"Name": "salary_max", "Type": "double"},
            {"Name": "salary_mid", "Type": "double"},
            {"Name": "posted_date", "Type": "timestamp"},
            {"Name": "extracted_skills", "Type": "array<string>"},
            {"Name": "skill_count", "Type": "int"},
            {"Name": "source", "Type": "string"},
            {"Name": "ingested_at", "Type": "string"},
        ]
        job_skills_columns = [
            {"Name": "job_id", "Type": "string"},
            {"Name": "posted_date", "Type": "timestamp"},
            {"Name": "salary_mid", "Type": "double"},
            {"Name": "skill", "Type": "string"},
            {"Name": "country", "Type": "string"},
        ]
        partition_keys = [
            {"Name": "year", "Type": "int"},
            {"Name": "month", "Type": "int"},
        ]
        serde = {
            "SerializationLibrary": "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
        }

        for table_name, columns, location in [
            ("jobs_kaggle", jobs_columns, f"s3://{self.bucket}/processed/jobs_kaggle/"),
            ("job_skills_kaggle", job_skills_columns, f"s3://{self.bucket}/processed/job_skills_kaggle/"),
        ]:
            try:
                glue.create_table(
                    DatabaseName=database,
                    TableInput={
                        "Name": table_name,
                        "StorageDescriptor": {
                            "Columns": columns,
                            "Location": location,
                            "InputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
                            "OutputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat",
                            "SerdeInfo": serde,
                        },
                        "PartitionKeys": partition_keys,
                        "TableType": "EXTERNAL_TABLE",
                    },
                )
                logger.info("Created %s table", table_name)
            except glue.exceptions.AlreadyExistsException:
                logger.info("%s table already exists", table_name)

        unified_view_sql = """
        CREATE OR REPLACE VIEW job_market_db.job_skills_all AS
        SELECT * FROM job_market_db.job_skills
        UNION ALL
        SELECT * FROM job_market_db.job_skills_kaggle;
        """
        logger.info("Unified view SQL (run in Athena):\n%s", unified_view_sql)

    def run(self, dataset_key: str = "primary") -> Dict:
        """Full pipeline: download -> process -> upload -> register."""
        logger.info("=" * 60)
        logger.info("KAGGLE DATA LOADING PIPELINE")
        logger.info("=" * 60)

        data_dir = self.download_dataset(dataset_key)

        try:
            jobs_df = self.process_linkedin_jobs(data_dir)
            job_skills_df = self.create_job_skills_table(jobs_df)
            result = self.upload_to_s3(jobs_df, job_skills_df)
            self.register_in_glue_catalog()
        finally:
            shutil.rmtree(data_dir, ignore_errors=True)

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info("Jobs loaded: %s", f"{result['jobs_count']:,}")
        logger.info("Job-skill records: %s", f"{result['job_skills_count']:,}")
        logger.info("Data location: s3://%s/processed/", self.bucket)

        return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load Kaggle job data to S3")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument(
        "--dataset",
        default="primary",
        choices=["primary", "secondary", "data_science"],
    )
    parser.add_argument("--region", default="us-east-1")

    args = parser.parse_args()

    loader = KaggleDataLoader(bucket=args.bucket, region=args.region)
    result = loader.run(dataset_key=args.dataset)

    print(f"\nData loaded! Next steps:")
    print(f"   1. Repair partitions: MSCK REPAIR TABLE job_market_db.job_skills_kaggle")
    print(f"   2. Train model: python -m ml.training.train --bucket {args.bucket} --data-source kaggle")
