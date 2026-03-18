"""
Adzuna API Job Scraper

Production-grade scraper for collecting job postings from Adzuna API.
Supports multiple countries, pagination, and skill extraction.

Usage:
    # Set environment variables
    export ADZUNA_APP_ID="your_app_id"
    export ADZUNA_APP_KEY="your_app_key"
    
    # Run scraper
    python -m ingestion.scrapers.adzuna_scraper --bucket your-bucket --countries us,gb

Prerequisites:
    pip install requests pandas pyarrow boto3 awswrangler tenacity
    
    # Get API credentials at: https://developer.adzuna.com/
"""

import os
import re
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import numpy as np
import boto3
import awswrangler as wr
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Adzuna API configuration
ADZUNA_BASE_URL = "https://api.adzuna.com/v1/api"

# Supported countries with their codes
ADZUNA_COUNTRIES = {
    "us": "United States",
    "gb": "United Kingdom",
    "ca": "Canada",
    "au": "Australia",
    "de": "Germany",
    "fr": "France",
    "in": "India",
    "nl": "Netherlands",
    "nz": "New Zealand",
    "pl": "Poland",
    "sg": "Singapore",
    "za": "South Africa",
}

# Tech job categories to search
TECH_CATEGORIES = [
    "it-jobs",
    "engineering-jobs",
    "scientific-qa-jobs",
]

# Search terms to maximize tech job coverage
TECH_SEARCH_TERMS = [
    "software engineer",
    "data scientist",
    "data engineer",
    "machine learning",
    "devops",
    "cloud engineer",
    "frontend developer",
    "backend developer",
    "full stack developer",
    "python developer",
    "java developer",
    "data analyst",
    "product manager tech",
    "cybersecurity",
    "AI engineer",
    "platform engineer",
    "site reliability",
    "database administrator",
    "solutions architect",
]

# Skills to extract from job descriptions
TECH_SKILLS = {
    # Programming languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "golang",
    "rust", "ruby", "php", "scala", "kotlin", "swift", "r", "sql", "bash",
    # Frameworks & libraries
    "react", "angular", "vue", "vuejs", "node", "nodejs", "django", "flask",
    "fastapi", "spring", "springboot", "express", "rails", "nextjs", "nuxt",
    ".net", "dotnet", "laravel", "tensorflow", "pytorch", "keras", "scikit-learn",
    "pandas", "numpy", "spark", "pyspark",
    # Databases
    "postgresql", "postgres", "mysql", "mongodb", "redis", "elasticsearch",
    "cassandra", "dynamodb", "snowflake", "bigquery", "redshift", "oracle",
    "sql server", "sqlite", "neo4j", "cockroachdb",
    # Cloud & infrastructure
    "aws", "azure", "gcp", "google cloud", "kubernetes", "k8s", "docker",
    "terraform", "ansible", "jenkins", "circleci", "github actions", "gitlab",
    "cloudformation", "pulumi", "helm", "istio", "serverless", "lambda",
    # Data & ML
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "mlops", "llm", "langchain", "openai", "huggingface",
    "airflow", "kafka", "flink", "dbt", "hadoop", "hive", "presto", "trino",
    "databricks", "sagemaker", "mlflow", "kubeflow", "ray",
    # Tools & practices
    "git", "linux", "unix", "agile", "scrum", "ci/cd", "rest", "graphql",
    "microservices", "api", "jira", "confluence",
}


class AdzunaScraper:
    """Production-grade Adzuna API scraper with rate limiting and error handling."""

    def __init__(
        self,
        app_id: Optional[str] = None,
        app_key: Optional[str] = None,
        bucket: Optional[str] = None,
        region: str = "us-east-1",
    ):
        self.app_id = app_id or os.environ.get("ADZUNA_APP_ID")
        self.app_key = app_key or os.environ.get("ADZUNA_APP_KEY")
        self.bucket = bucket
        self.region = region
        
        if not self.app_id or not self.app_key:
            raise ValueError(
                "Adzuna API credentials required. Set ADZUNA_APP_ID and ADZUNA_APP_KEY "
                "environment variables or pass them to the constructor.\n"
                "Get your free API key at: https://developer.adzuna.com/"
            )
        
        self.session = requests.Session()
        self.s3 = boto3.client("s3", region_name=region) if bucket else None
        self._seen_job_ids: Set[str] = set()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _make_request(self, endpoint: str, params: Dict) -> Dict:
        """Make API request with retry logic."""
        params["app_id"] = self.app_id
        params["app_key"] = self.app_key
        
        url = f"{ADZUNA_BASE_URL}/{endpoint}"
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        return response.json()

    def search_jobs(
        self,
        country: str = "us",
        what: Optional[str] = None,
        where: Optional[str] = None,
        category: Optional[str] = None,
        page: int = 1,
        results_per_page: int = 50,
        max_days_old: int = 30,
        salary_min: Optional[int] = None,
    ) -> Dict:
        """
        Search for jobs using Adzuna API.
        
        Args:
            country: Country code (us, gb, ca, etc.)
            what: Search keywords
            where: Location
            category: Job category (it-jobs, engineering-jobs, etc.)
            page: Page number (1-indexed)
            results_per_page: Results per page (max 50)
            max_days_old: Maximum age of job posting in days
            salary_min: Minimum salary filter
            
        Returns:
            API response with job results
        """
        endpoint = f"jobs/{country}/search/{page}"
        
        params = {
            "results_per_page": min(results_per_page, 50),
            "content-type": "application/json",
            "max_days_old": max_days_old,
        }
        
        if what:
            params["what"] = what
        if where:
            params["where"] = where
        if category:
            params["category"] = category
        if salary_min:
            params["salary_min"] = salary_min
            
        return self._make_request(endpoint, params)

    def get_categories(self, country: str = "us") -> List[Dict]:
        """Get available job categories for a country."""
        endpoint = f"jobs/{country}/categories"
        response = self._make_request(endpoint, {})
        return response.get("results", [])

    def get_salary_histogram(
        self,
        country: str = "us",
        what: Optional[str] = None,
        where: Optional[str] = None,
    ) -> Dict:
        """Get salary distribution for a job search."""
        endpoint = f"jobs/{country}/histogram"
        params = {}
        if what:
            params["what"] = what
        if where:
            params["where"] = where
        return self._make_request(endpoint, params)

    @staticmethod
    def extract_skills(text: str) -> List[str]:
        """Extract tech skills from job description."""
        if not text:
            return []
        
        text_lower = text.lower()
        found: Set[str] = set()
        
        for skill in TECH_SKILLS:
            # Word boundary matching
            pattern = r"\b" + re.escape(skill) + r"\b"
            if re.search(pattern, text_lower):
                # Normalize skill names
                normalized = skill.replace(" ", "_").replace(".", "").replace("/", "_")
                found.add(normalized)
        
        return sorted(found)

    @staticmethod
    def generate_job_id(job: Dict) -> str:
        """Generate a unique job ID from job data."""
        # Use Adzuna's ID if available, otherwise hash key fields
        if "id" in job:
            return f"adzuna_{job['id']}"
        
        key = f"{job.get('title', '')}-{job.get('company', {}).get('display_name', '')}-{job.get('location', {}).get('display_name', '')}"
        return f"adzuna_{hashlib.md5(key.encode()).hexdigest()[:12]}"

    def parse_job(self, job: Dict, country: str) -> Optional[Dict]:
        """Parse a single job posting into normalized format."""
        job_id = self.generate_job_id(job)
        
        # Skip duplicates
        if job_id in self._seen_job_ids:
            return None
        self._seen_job_ids.add(job_id)
        
        # Extract location info
        location = job.get("location", {})
        area = location.get("area", [])
        
        # Parse salary
        salary_min = job.get("salary_min")
        salary_max = job.get("salary_max")
        salary_mid = None
        if salary_min and salary_max:
            salary_mid = (salary_min + salary_max) / 2
        elif salary_min:
            salary_mid = salary_min
        elif salary_max:
            salary_mid = salary_max
        
        # Extract skills from description
        description = job.get("description", "")
        title = job.get("title", "")
        skills = self.extract_skills(f"{title} {description}")
        
        # Parse date
        created_str = job.get("created", "")
        try:
            posted_date = pd.to_datetime(created_str)
        except Exception:
            posted_date = pd.Timestamp.now()
        
        return {
            "job_id": job_id,
            "title": title,
            "company": job.get("company", {}).get("display_name", ""),
            "description": description,
            "location": location.get("display_name", ""),
            "location_area": area,
            "country": country.upper(),
            "salary_min": salary_min,
            "salary_max": salary_max,
            "salary_mid": salary_mid,
            "salary_is_predicted": job.get("salary_is_predicted", 0),
            "contract_type": job.get("contract_type", ""),
            "contract_time": job.get("contract_time", ""),
            "category": job.get("category", {}).get("label", ""),
            "category_tag": job.get("category", {}).get("tag", ""),
            "redirect_url": job.get("redirect_url", ""),
            "posted_date": posted_date,
            "latitude": job.get("latitude"),
            "longitude": job.get("longitude"),
            "extracted_skills": skills,
            "skill_count": len(skills),
            "source": "adzuna",
            "ingested_at": datetime.utcnow().isoformat(),
        }

    def scrape_country(
        self,
        country: str = "us",
        max_pages_per_term: int = 20,
        max_days_old: int = 30,
    ) -> pd.DataFrame:
        """
        Scrape all tech jobs for a country.
        
        Args:
            country: Country code
            max_pages_per_term: Max pages to fetch per search term
            max_days_old: Maximum job age in days
            
        Returns:
            DataFrame with all jobs
        """
        logger.info(f"Scraping {ADZUNA_COUNTRIES.get(country, country)}...")
        all_jobs = []
        
        # Search by category first
        for category in TECH_CATEGORIES:
            logger.info(f"  Category: {category}")
            page = 1
            
            while page <= max_pages_per_term:
                try:
                    response = self.search_jobs(
                        country=country,
                        category=category,
                        page=page,
                        max_days_old=max_days_old,
                    )
                    
                    results = response.get("results", [])
                    if not results:
                        break
                    
                    for job in results:
                        parsed = self.parse_job(job, country)
                        if parsed:
                            all_jobs.append(parsed)
                    
                    total = response.get("count", 0)
                    logger.info(f"    Page {page}: {len(results)} jobs (total available: {total})")
                    
                    if len(results) < 50:
                        break
                    page += 1
                    
                except Exception as e:
                    logger.warning(f"    Error on page {page}: {e}")
                    break
        
        # Search by keywords for broader coverage
        for term in TECH_SEARCH_TERMS:
            logger.info(f"  Search term: {term}")
            page = 1
            
            while page <= max_pages_per_term:
                try:
                    response = self.search_jobs(
                        country=country,
                        what=term,
                        page=page,
                        max_days_old=max_days_old,
                    )
                    
                    results = response.get("results", [])
                    if not results:
                        break
                    
                    for job in results:
                        parsed = self.parse_job(job, country)
                        if parsed:
                            all_jobs.append(parsed)
                    
                    if len(results) < 50:
                        break
                    page += 1
                    
                except Exception as e:
                    logger.warning(f"    Error on page {page}: {e}")
                    break
        
        logger.info(f"  Total unique jobs for {country}: {len(all_jobs)}")
        return pd.DataFrame(all_jobs)

    def scrape_all_countries(
        self,
        countries: Optional[List[str]] = None,
        max_pages_per_term: int = 20,
        max_days_old: int = 30,
    ) -> pd.DataFrame:
        """Scrape jobs from multiple countries."""
        if countries is None:
            countries = ["us", "gb", "ca"]
        
        logger.info("=" * 60)
        logger.info("ADZUNA JOB SCRAPING PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Countries: {countries}")
        logger.info(f"Max days old: {max_days_old}")
        
        all_dfs = []
        for country in countries:
            if country not in ADZUNA_COUNTRIES:
                logger.warning(f"Unknown country code: {country}")
                continue
            
            df = self.scrape_country(
                country=country,
                max_pages_per_term=max_pages_per_term,
                max_days_old=max_days_old,
            )
            all_dfs.append(df)
        
        combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
        logger.info(f"Total jobs scraped: {len(combined)}")
        
        return combined

    def create_job_skills_table(self, jobs_df: pd.DataFrame) -> pd.DataFrame:
        """Explode extracted_skills into a junction table."""
        logger.info("Creating job_skills junction table...")
        
        # Select relevant columns
        skills_df = jobs_df[
            ["job_id", "posted_date", "salary_mid", "country", "extracted_skills"]
        ].copy()
        
        # Add year/month for partitioning
        skills_df["year"] = skills_df["posted_date"].dt.year
        skills_df["month"] = skills_df["posted_date"].dt.month
        
        # Explode skills array
        skills_df = skills_df.explode("extracted_skills")
        skills_df = skills_df.rename(columns={"extracted_skills": "skill"})
        skills_df = skills_df.dropna(subset=["skill"])
        skills_df = skills_df[skills_df["skill"] != ""]
        
        # Select final columns
        final_cols = ["job_id", "posted_date", "salary_mid", "skill", "country", "year", "month"]
        skills_df = skills_df[final_cols]
        
        logger.info(f"Created {len(skills_df):,} job-skill records")
        logger.info(f"Unique skills: {skills_df['skill'].nunique():,}")
        
        return skills_df

    def upload_to_s3(
        self,
        jobs_df: pd.DataFrame,
        job_skills_df: pd.DataFrame,
    ) -> Dict:
        """Upload data to S3 in Parquet format."""
        if not self.bucket:
            raise ValueError("S3 bucket not configured")
        
        logger.info(f"Uploading to S3 bucket: {self.bucket}")
        
        # Add partitioning columns to jobs
        jobs_df["year"] = jobs_df["posted_date"].dt.year
        jobs_df["month"] = jobs_df["posted_date"].dt.month
        
        # Upload raw jobs
        raw_path = f"s3://{self.bucket}/raw/jobs/source=adzuna/"
        logger.info(f"Writing raw jobs to: {raw_path}")
        wr.s3.to_parquet(
            df=jobs_df,
            path=raw_path,
            dataset=True,
            partition_cols=["year", "month"],
            mode="overwrite",
        )
        
        # Upload processed jobs
        processed_jobs_path = f"s3://{self.bucket}/processed/jobs/"
        logger.info(f"Writing processed jobs to: {processed_jobs_path}")
        wr.s3.to_parquet(
            df=jobs_df,
            path=processed_jobs_path,
            dataset=True,
            partition_cols=["year", "month"],
            mode="overwrite",
        )
        
        # Upload job_skills
        job_skills_path = f"s3://{self.bucket}/processed/job_skills/"
        logger.info(f"Writing job_skills to: {job_skills_path}")
        wr.s3.to_parquet(
            df=job_skills_df,
            path=job_skills_path,
            dataset=True,
            partition_cols=["year", "month"],
            mode="overwrite",
        )
        
        # Upload ML training data
        ml_path = f"s3://{self.bucket}/ml/training_data/"
        logger.info(f"Writing ML training data to: {ml_path}")
        wr.s3.to_parquet(
            df=job_skills_df,
            path=ml_path,
            dataset=True,
            mode="overwrite",
        )
        
        logger.info("All data uploaded to S3")
        
        return {
            "raw_path": raw_path,
            "processed_jobs_path": processed_jobs_path,
            "job_skills_path": job_skills_path,
            "ml_path": ml_path,
            "jobs_count": len(jobs_df),
            "job_skills_count": len(job_skills_df),
        }

    def run(
        self,
        countries: Optional[List[str]] = None,
        max_pages_per_term: int = 20,
        max_days_old: int = 30,
        upload: bool = True,
    ) -> Dict:
        """Run the complete scraping pipeline."""
        # Scrape jobs
        jobs_df = self.scrape_all_countries(
            countries=countries,
            max_pages_per_term=max_pages_per_term,
            max_days_old=max_days_old,
        )
        
        if jobs_df.empty:
            logger.error("No jobs scraped!")
            return {"jobs_count": 0, "job_skills_count": 0}
        
        # Create job_skills table
        job_skills_df = self.create_job_skills_table(jobs_df)
        
        # Upload to S3
        result = {"jobs_count": len(jobs_df), "job_skills_count": len(job_skills_df)}
        
        if upload and self.bucket:
            upload_result = self.upload_to_s3(jobs_df, job_skills_df)
            result.update(upload_result)
        else:
            # Save locally
            os.makedirs("data", exist_ok=True)
            jobs_df.to_parquet("data/jobs_adzuna.parquet", index=False)
            job_skills_df.to_parquet("data/job_skills_adzuna.parquet", index=False)
            logger.info("Data saved locally to data/ directory")
        
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Jobs scraped: {result['jobs_count']:,}")
        logger.info(f"Job-skill records: {result['job_skills_count']:,}")
        
        return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape jobs from Adzuna API")
    parser.add_argument("--bucket", help="S3 bucket name")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument(
        "--countries",
        default="us,gb,ca",
        help="Comma-separated country codes (us,gb,ca,au,de,etc.)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=20,
        help="Max pages per search term",
    )
    parser.add_argument(
        "--max-days-old",
        type=int,
        default=30,
        help="Maximum job age in days",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Save locally instead of uploading to S3",
    )
    
    args = parser.parse_args()
    
    scraper = AdzunaScraper(
        bucket=args.bucket,
        region=args.region,
    )
    
    result = scraper.run(
        countries=args.countries.split(","),
        max_pages_per_term=args.max_pages,
        max_days_old=args.max_days_old,
        upload=not args.no_upload and args.bucket is not None,
    )
    
    print(f"\nScraping complete!")
    print(f"Jobs: {result['jobs_count']:,}")
    print(f"Job-skill records: {result['job_skills_count']:,}")
    
    if args.bucket:
        print(f"\nNext steps:")
        print(f"  1. Train model: python -m ml.training.train --bucket {args.bucket}")
