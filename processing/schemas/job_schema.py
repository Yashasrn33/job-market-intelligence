from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class JobLocation:
    display_name: str
    area: List[str]
    country: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None


@dataclass
class JobSalary:
    min: Optional[float] = None
    max: Optional[float] = None
    currency: str = "LOCAL"
    is_predicted: bool = False


@dataclass
class JobRecord:
    job_id: str
    source: str
    title: str
    company: str
    description: str
    location: JobLocation
    salary: JobSalary
    posted_date: str
    url: str
    ingested_at: str

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "JobRecord":
        loc = d.get("location") or {}
        sal = d.get("salary") or {}
        return JobRecord(
            job_id=d.get("job_id", ""),
            source=d.get("source", ""),
            title=d.get("title", ""),
            company=d.get("company", ""),
            description=d.get("description", ""),
            location=JobLocation(
                display_name=loc.get("display_name", ""),
                area=list(loc.get("area") or []),
                country=loc.get("country", ""),
                latitude=loc.get("latitude"),
                longitude=loc.get("longitude"),
            ),
            salary=JobSalary(
                min=sal.get("min"),
                max=sal.get("max"),
                currency=sal.get("currency", "LOCAL"),
                is_predicted=bool(sal.get("is_predicted", False)),
            ),
            posted_date=d.get("posted_date", ""),
            url=d.get("url", ""),
            ingested_at=d.get("ingested_at", ""),
        )

