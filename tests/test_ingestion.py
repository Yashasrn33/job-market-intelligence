from ingestion.scrapers.adzuna import transform_job


def test_transform_job_basic_fields():
    raw = {
        "id": 123,
        "title": "Data Engineer",
        "company": {"display_name": "Co"},
        "description": "SQL and Python",
        "location": {"display_name": "London", "area": ["UK", "London"]},
    }
    out = transform_job(raw, country="gb", query="data engineer")
    assert out["source"] == "adzuna"
    assert out["title"] == "Data Engineer"
    assert out["company"] == "Co"
    assert out["location"]["country"] == "GB"
    assert out["salary"]["currency"] == "GBP"
    assert out["job_id"]  # deterministic hash


def test_transform_job_us_currency():
    raw = {"id": 1, "title": "SWE", "company": {"display_name": "X"}}
    out = transform_job(raw, country="us", query="swe")
    assert out["salary"]["currency"] == "USD"


def test_transform_job_missing_salary():
    raw = {"id": 2, "title": "PM", "company": {"display_name": "Y"}}
    out = transform_job(raw, country="ca", query="pm")
    assert out["salary"]["min"] is None
    assert out["salary"]["max"] is None
