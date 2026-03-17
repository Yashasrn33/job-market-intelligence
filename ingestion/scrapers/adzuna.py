import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Generator, Iterable, List, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdzunaConfig:
    base_url: str = "https://api.adzuna.com/v1/api/jobs"
    countries: Iterable[str] = ("us", "gb", "ca", "au", "de")
    search_queries: Iterable[str] = (
        "software engineer",
        "data scientist",
        "machine learning",
        "python developer",
        "cloud engineer",
        "devops",
        "frontend developer",
        "backend developer",
        "data engineer",
        "AI engineer",
        "full stack developer",
        "cybersecurity",
        "product manager tech",
    )
    results_per_page: int = 50
    pages_per_query_country: int = 3
    max_days_old: int = 7
    request_timeout_s: int = 30


def _generate_job_id(job: Dict) -> str:
    unique_str = (
        f"{job.get('id', '')}-"
        f"{job.get('title', '')}-"
        f"{job.get('company', {}).get('display_name', '')}"
    )
    return hashlib.md5(unique_str.encode("utf-8")).hexdigest()


def _currency_for_country(country: str) -> str:
    if country == "us":
        return "USD"
    if country == "gb":
        return "GBP"
    return "LOCAL"


def transform_job(job: Dict, country: str, query: str, ingested_at: Optional[str] = None) -> Dict:
    """Transform Adzuna job to a normalized JSON record (NDJSON-friendly)."""
    return {
        "job_id": _generate_job_id(job),
        "source": "adzuna",
        "source_id": str(job.get("id", "")),
        "title": job.get("title", ""),
        "company": job.get("company", {}).get("display_name", ""),
        "description": job.get("description", ""),
        "location": {
            "display_name": job.get("location", {}).get("display_name", ""),
            "area": job.get("location", {}).get("area", []),
            "country": country.upper(),
            "latitude": job.get("latitude"),
            "longitude": job.get("longitude"),
        },
        "salary": {
            "min": job.get("salary_min"),
            "max": job.get("salary_max"),
            "is_predicted": job.get("salary_is_predicted", 0) == 1,
            "currency": _currency_for_country(country),
        },
        "contract": {
            "type": job.get("contract_type"),
            "time": job.get("contract_time"),
        },
        "category": job.get("category", {}).get("label", ""),
        "url": job.get("redirect_url", ""),
        "posted_date": job.get("created", ""),
        "search_query": query,
        "ingested_at": ingested_at or datetime.utcnow().isoformat(),
    }


class AdzunaScraper:
    def __init__(self, app_id: str, app_key: str, config: Optional[AdzunaConfig] = None):
        self.app_id = app_id
        self.app_key = app_key
        self.config = config or AdzunaConfig()

    def fetch_page(self, country: str, query: str, page: int) -> Dict:
        url = f"{self.config.base_url}/{country}/search/{page}"
        params = {
            "app_id": self.app_id,
            "app_key": self.app_key,
            "results_per_page": self.config.results_per_page,
            "what": query,
            "content-type": "application/json",
            "sort_by": "date",
            "max_days_old": self.config.max_days_old,
        }
        response = requests.get(url, params=params, timeout=self.config.request_timeout_s)
        response.raise_for_status()
        return response.json()

    def scrape(self) -> Generator[Dict, None, None]:
        ingested_at = datetime.utcnow().isoformat()
        for country in self.config.countries:
            for query in self.config.search_queries:
                logger.info("Fetching: %s in %s", query, country)
                try:
                    for page in range(1, self.config.pages_per_query_country + 1):
                        data = self.fetch_page(country, query, page)
                        results: List[Dict] = data.get("results", []) or []
                        if not results:
                            break
                        for job in results:
                            yield transform_job(job, country, query, ingested_at=ingested_at)
                except requests.exceptions.RequestException as e:
                    logger.warning("Failed to fetch %s/%s: %s", query, country, e)
                    continue


def to_ndjson(records: Iterable[Dict]) -> str:
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in records)

