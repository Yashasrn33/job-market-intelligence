"""
AWS Glue ETL Job: Process raw jobs, extract skills, write Parquet
Run with: --S3_BUCKET <bucket> --DATABASE <glue_db>
"""
import sys
import re
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.window import Window

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)

args = getResolvedOptions(sys.argv, ['JOB_NAME', 'S3_BUCKET', 'DATABASE'])
S3_BUCKET = args['S3_BUCKET']
DATABASE = args['DATABASE']
job.init(args['JOB_NAME'], args)

TECH_SKILLS = {
    'python', 'java', 'javascript', 'typescript', 'c++', 'go', 'rust',
    'react', 'angular', 'vue', 'nodejs', 'django', 'flask', 'fastapi',
    'postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch',
    'aws', 'azure', 'gcp', 'kubernetes', 'docker', 'terraform',
    'spark', 'airflow', 'kafka', 'dbt',
    'machine learning', 'deep learning', 'tensorflow', 'pytorch',
    'ci/cd', 'jenkins', 'github actions',
}


def extract_skills(text):
    if not text:
        return []
    text_lower = text.lower()
    return [s for s in TECH_SKILLS if re.search(r'\b' + re.escape(s) + r'\b', text_lower)]


extract_skills_udf = F.udf(extract_skills, ArrayType(StringType()))

# Read raw NDJSON
raw_df = spark.read.json(f"s3://{S3_BUCKET}/raw/jobs/source=adzuna/")

# Deduplicate (keep latest ingestion per job_id)
window = Window.partitionBy("job_id").orderBy(F.col("ingested_at").desc())
deduped_df = (
    raw_df
    .withColumn("rn", F.row_number().over(window))
    .filter("rn = 1")
    .drop("rn")
)

# Transform
processed_df = (
    deduped_df
    .filter(F.col("description").isNotNull())
    .withColumn("extracted_skills", extract_skills_udf(F.col("description")))
    .withColumn("skill_count", F.size("extracted_skills"))
    .withColumn("salary_min_usd", F.col("salary.min"))
    .withColumn("salary_max_usd", F.col("salary.max"))
    .withColumn("salary_mid_usd", (F.col("salary.min") + F.col("salary.max")) / 2)
    .withColumn("posted_date", F.to_date("posted_date"))
    .withColumn("year", F.year("posted_date"))
    .withColumn("month", F.month("posted_date"))
)

# --- Write processed jobs table ---
jobs_output = processed_df.select(
    "job_id", "title", "company", "description",
    F.col("location.display_name").alias("location"),
    F.col("location.country").alias("country"),
    "salary_min_usd", "salary_max_usd", "salary_mid_usd",
    "category", "url", "posted_date",
    "extracted_skills", "skill_count",
    "year", "month",
)

jobs_output.write.mode("overwrite") \
    .partitionBy("year", "month") \
    .parquet(f"s3://{S3_BUCKET}/processed/jobs/")

# --- Write job_skills junction table (exploded) ---
job_skills = processed_df.select(
    "job_id", "posted_date",
    F.col("location.country").alias("country"),
    "salary_mid_usd",
    F.explode("extracted_skills").alias("skill"),
    "year", "month",
)

job_skills.write.mode("overwrite") \
    .partitionBy("year", "month") \
    .parquet(f"s3://{S3_BUCKET}/processed/job_skills/")

print(f"Processed {jobs_output.count()} jobs")
job.commit()
