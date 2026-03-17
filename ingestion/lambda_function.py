"""
AWS Lambda: Adzuna Job Scraper
Fetches job postings and stores them in S3 data lake.
"""
import json
import boto3
import requests
from datetime import datetime
import hashlib
import os

s3 = boto3.client('s3')
secrets = boto3.client('secretsmanager')

BUCKET_NAME = os.environ.get('S3_BUCKET', 'job-market-intelligence')

SEARCH_QUERIES = [
    'software engineer', 'data scientist', 'machine learning',
    'python developer', 'cloud engineer', 'devops', 'data engineer',
    'AI engineer', 'full stack developer', 'backend developer',
]

COUNTRIES = ['us', 'gb', 'ca', 'au', 'de']


def get_api_credentials():
    response = secrets.get_secret_value(SecretId='job-market/adzuna-api')
    return json.loads(response['SecretString'])


def fetch_jobs(app_id, app_key, country, query, page=1):
    url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"
    params = {
        'app_id': app_id, 'app_key': app_key,
        'results_per_page': 50, 'what': query,
        'content-type': 'application/json',
        'sort_by': 'date', 'max_days_old': 7,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def transform_job(job, country, query):
    job_id = hashlib.md5(
        f"{job.get('id')}-{job.get('title')}".encode()
    ).hexdigest()
    return {
        'job_id': job_id,
        'source': 'adzuna',
        'source_id': str(job.get('id', '')),
        'title': job.get('title', ''),
        'company': job.get('company', {}).get('display_name', ''),
        'description': job.get('description', ''),
        'location': {
            'display_name': job.get('location', {}).get('display_name', ''),
            'country': country.upper(),
            'latitude': job.get('latitude'),
            'longitude': job.get('longitude'),
        },
        'salary': {
            'min': job.get('salary_min'),
            'max': job.get('salary_max'),
            'is_predicted': job.get('salary_is_predicted', 0) == 1,
        },
        'category': job.get('category', {}).get('label', ''),
        'url': job.get('redirect_url', ''),
        'posted_date': job.get('created', ''),
        'search_query': query,
        'ingested_at': datetime.utcnow().isoformat(),
    }


def lambda_handler(event, context):
    execution_date = datetime.utcnow().strftime('%Y-%m-%d')
    creds = get_api_credentials()

    seen_ids, jobs = set(), []

    for country in COUNTRIES:
        for query in SEARCH_QUERIES:
            try:
                for page in range(1, 3):
                    data = fetch_jobs(
                        creds['app_id'], creds['app_key'],
                        country, query, page,
                    )
                    for job in data.get('results', []):
                        transformed = transform_job(job, country, query)
                        if transformed['job_id'] not in seen_ids:
                            seen_ids.add(transformed['job_id'])
                            jobs.append(transformed)
            except Exception as e:
                print(f"Error {query}/{country}: {e}")

    key = f"raw/jobs/source=adzuna/date={execution_date}/jobs.json"
    body = '\n'.join(json.dumps(job) for job in jobs)
    s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=body.encode('utf-8'))

    return {
        'statusCode': 200,
        'body': {'jobs_collected': len(jobs), 's3_key': key},
    }
