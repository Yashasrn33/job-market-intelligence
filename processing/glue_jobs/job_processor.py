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

# Read raw data — supports both Parquet and JSON
raw_path = f"s3://{S3_BUCKET}/raw/jobs/source=adzuna/"
try:
    raw_df = spark.read.parquet(raw_path)
    print("Reading data as Parquet")
except:
    raw_df = spark.read.json(raw_path)
    print("Reading data as JSON")

print(f"Loaded {raw_df.count()} raw records")
print(f"Columns: {raw_df.columns}")

# Synthesize job_id if missing
if "job_id" not in raw_df.columns:
    concat_cols = F.concat_ws("||", *[F.col(c).cast("string") for c in raw_df.columns])
    raw_df = raw_df.withColumn("job_id", F.sha2(concat_cols, 256))

# Choose ordering column for deduplication
if "ingested_at" in raw_df.columns:
    order_col = F.col("ingested_at").desc_nulls_last()
elif "posted_date" in raw_df.columns:
    order_col = F.col("posted_date").desc_nulls_last()
else:
    order_col = F.lit(1)

# Deduplicate (keep latest record per job_id)
window = Window.partitionBy("job_id").orderBy(order_col)
deduped_df = (
    raw_df
    .withColumn("rn", F.row_number().over(window))
    .filter("rn = 1")
    .drop("rn")
)

cols = set(deduped_df.columns)
processed_df = deduped_df

# --- Handle location flexibly (struct OR string) ---
if "location" in cols:
    # Check if location is a struct or string
    location_type = str(deduped_df.schema["location"].dataType)
    if "StructType" in location_type:
        # Nested struct format (from JSON ingestion)
        processed_df = processed_df.withColumn("location_display_name", F.col("location.display_name"))
        processed_df = processed_df.withColumn("location_country", F.col("location.country"))
    else:
        # Flat string format (from Parquet ingestion)
        processed_df = processed_df.withColumn("location_display_name", F.col("location"))
        # Use existing country column if available
        if "country" in cols:
            processed_df = processed_df.withColumn("location_country", F.col("country"))
        else:
            processed_df = processed_df.withColumn("location_country", F.lit(None).cast(StringType()))
else:
    processed_df = processed_df.withColumn("location_display_name", F.lit(None).cast(StringType()))
    processed_df = processed_df.withColumn("location_country", F.col("country") if "country" in cols else F.lit(None).cast(StringType()))

# --- Handle description and skill extraction ---
if "description" in cols:
    processed_df = processed_df.filter(F.col("description").isNotNull())
    # Check if skills already extracted
    if "extracted_skills" in cols:
        print("Using pre-extracted skills")
    else:
        processed_df = processed_df.withColumn("extracted_skills", extract_skills_udf(F.col("description")))
else:
    processed_df = processed_df.withColumn("description", F.lit(None).cast(StringType()))
    processed_df = processed_df.withColumn("extracted_skills", F.array().cast(ArrayType(StringType())))

# Ensure skill_count exists
if "skill_count" not in processed_df.columns:
    processed_df = processed_df.withColumn("skill_count", F.size(F.col("extracted_skills")))

# --- Handle salary flexibly (struct OR flat columns) ---
if "salary" in cols:
    salary_type = str(deduped_df.schema["salary"].dataType)
    if "StructType" in salary_type:
        processed_df = processed_df.withColumn("salary_min_usd", F.col("salary.min"))
        processed_df = processed_df.withColumn("salary_max_usd", F.col("salary.max"))
    else:
        processed_df = processed_df.withColumn("salary_min_usd", F.col("salary").cast("double"))
        processed_df = processed_df.withColumn("salary_max_usd", F.col("salary").cast("double"))
elif "salary_min" in cols:
    # Already flat columns
    processed_df = processed_df.withColumn("salary_min_usd", F.col("salary_min"))
    processed_df = processed_df.withColumn("salary_max_usd", F.col("salary_max"))
else:
    processed_df = processed_df.withColumn("salary_min_usd", F.lit(None).cast("double"))
    processed_df = processed_df.withColumn("salary_max_usd", F.lit(None).cast("double"))

# Calculate midpoint
processed_df = processed_df.withColumn(
    "salary_mid_usd",
    (F.col("salary_min_usd") + F.col("salary_max_usd")) / 2,
)

# --- Handle posted_date and partitions ---
if "posted_date" in cols:
    processed_df = processed_df.withColumn("posted_date", F.to_date("posted_date"))
elif "date" in cols:
    processed_df = processed_df.withColumn("posted_date", F.to_date("date"))
else:
    processed_df = processed_df.withColumn("posted_date", F.current_date())

processed_df = processed_df.withColumn("year", F.year("posted_date"))
processed_df = processed_df.withColumn("month", F.month("posted_date"))

# --- Ensure optional columns exist ---
for name, dtype in [("title", StringType()), ("company", StringType()), 
                    ("category", StringType()), ("url", StringType())]:
    if name not in processed_df.columns:
        # Check for alternate column names
        if name == "url" and "redirect_url" in processed_df.columns:
            processed_df = processed_df.withColumn(name, F.col("redirect_url"))
        else:
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

jobs_count = jobs_output.count()
skills_count = job_skills.count()
print(f"✅ Processed {jobs_count} jobs with {skills_count} skill records")
job.commit()