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

# Some historical runs may not have a materialized job_id column.
# In that case, synthesize a stable surrogate key so the job doesn't fail.
if "job_id" not in raw_df.columns:
    concat_cols = F.concat_ws(
        "||",
        *[F.col(c).cast("string") for c in raw_df.columns]
    )
    raw_df = raw_df.withColumn("job_id", F.sha2(concat_cols, 256))

# Choose an ordering column for deduplication, preferring ingested_at if present,
# otherwise falling back to a generic date column if available.
if "ingested_at" in raw_df.columns:
    order_col = F.col("ingested_at").desc_nulls_last()
elif "date" in raw_df.columns:
    order_col = F.col("date").desc_nulls_last()
else:
    order_col = F.lit(1)

# Deduplicate (keep latest record per job_id based on chosen ordering)
window = Window.partitionBy("job_id").orderBy(order_col)
deduped_df = (
    raw_df
    .withColumn("rn", F.row_number().over(window))
    .filter("rn = 1")
    .drop("rn")
)

# Transform with defensive handling for missing columns
cols = set(deduped_df.columns)
processed_df = deduped_df

# Description and skills
if "description" in cols:
    processed_df = processed_df.filter(F.col("description").isNotNull())
    processed_df = processed_df.withColumn(
        "extracted_skills", extract_skills_udf(F.col("description"))
    )
else:
    processed_df = processed_df.withColumn(
        "description", F.lit(None).cast(StringType())
    )
    processed_df = processed_df.withColumn(
        "extracted_skills", F.array().cast(ArrayType(StringType()))
    )

processed_df = processed_df.withColumn(
    "skill_count", F.size(F.col("extracted_skills"))
)

# Salary fields
if "salary" in cols:
    processed_df = processed_df.withColumn("salary_min_usd", F.col("salary.min"))
    processed_df = processed_df.withColumn("salary_max_usd", F.col("salary.max"))
    processed_df = processed_df.withColumn(
        "salary_mid_usd",
        (F.col("salary.min") + F.col("salary.max")) / 2,
    )
else:
    processed_df = processed_df.withColumn("salary_min_usd", F.lit(None).cast("double"))
    processed_df = processed_df.withColumn("salary_max_usd", F.lit(None).cast("double"))
    processed_df = processed_df.withColumn("salary_mid_usd", F.lit(None).cast("double"))

# Posted date and partitions
if "posted_date" in cols:
    posted = F.to_date("posted_date")
elif "date" in cols:
    posted = F.to_date("date")
else:
    posted = F.current_date()

processed_df = processed_df.withColumn("posted_date", posted)
processed_df = processed_df.withColumn("year", F.year("posted_date"))
processed_df = processed_df.withColumn("month", F.month("posted_date"))

# Location helpers
if "location" in cols:
    processed_df = processed_df.withColumn(
        "location_display_name", F.col("location.display_name")
    )
    processed_df = processed_df.withColumn(
        "location_country", F.col("location.country")
    )
else:
    processed_df = processed_df.withColumn(
        "location_display_name", F.lit(None).cast(StringType())
    )
    processed_df = processed_df.withColumn(
        "location_country", F.lit(None).cast(StringType())
    )

# Ensure optional top-level columns exist so selects don't fail
for name, dtype in [
    ("title", StringType()),
    ("company", StringType()),
    ("category", StringType()),
    ("url", StringType()),
]:
    if name not in processed_df.columns:
        processed_df = processed_df.withColumn(name, F.lit(None).cast(dtype))

# --- Write processed jobs table ---
jobs_output = processed_df.select(
    "job_id", "title", "company", "description",
    F.col("location_display_name").alias("location"),
    F.col("location_country").alias("country"),
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
    F.col("location_country").alias("country"),
    "salary_mid_usd",
    F.explode("extracted_skills").alias("skill"),
    "year", "month",
)

job_skills.write.mode("overwrite") \
    .partitionBy("year", "month") \
    .parquet(f"s3://{S3_BUCKET}/processed/job_skills/")

print(f"Processed {jobs_output.count()} jobs")
job.commit()
