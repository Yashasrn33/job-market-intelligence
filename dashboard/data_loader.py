"""
Data loader for the Streamlit dashboard.

Supports two modes:
  1. AWS mode  – queries Athena views in job_market_db (requires credentials)
  2. Demo mode – generates realistic synthetic data so the dashboard works
                 out of the box without any AWS setup
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    import awswrangler as wr
    import boto3

    _HAS_AWS = True
except ImportError:
    _HAS_AWS = False


# ── skill universe (mirrors the Glue ETL + Athena views) ────────────────────

SKILL_CATEGORIES = {
    "python": "Languages", "java": "Languages", "javascript": "Languages",
    "typescript": "Languages", "c++": "Languages", "go": "Languages",
    "rust": "Languages", "ruby": "Languages", "scala": "Languages",
    "kotlin": "Languages", "swift": "Languages", "r": "Languages",
    "sql": "Languages", "bash": "Languages", "c#": "Languages", "php": "Languages",
    "react": "Frontend", "angular": "Frontend", "vue": "Frontend",
    "nextjs": "Frontend", "nuxt": "Frontend",
    "nodejs": "Backend", "django": "Backend", "flask": "Backend",
    "fastapi": "Backend", "spring": "Backend", "springboot": "Backend",
    "express": "Backend", "rails": "Backend", "laravel": "Backend",
    "dotnet": "Backend", "net": "Backend",
    "aws": "Cloud & Infra", "azure": "Cloud & Infra", "gcp": "Cloud & Infra",
    "kubernetes": "Cloud & Infra", "docker": "Cloud & Infra",
    "terraform": "Cloud & Infra", "ansible": "Cloud & Infra",
    "cloudformation": "Cloud & Infra", "helm": "Cloud & Infra",
    "serverless": "Cloud & Infra", "lambda": "Cloud & Infra",
    "postgresql": "Databases", "mysql": "Databases", "mongodb": "Databases",
    "redis": "Databases", "elasticsearch": "Databases", "dynamodb": "Databases",
    "snowflake": "Databases", "bigquery": "Databases", "redshift": "Databases",
    "oracle": "Databases", "sql_server": "Databases",
    "machine_learning": "ML & AI", "deep_learning": "ML & AI",
    "tensorflow": "ML & AI", "pytorch": "ML & AI", "scikit-learn": "ML & AI",
    "nlp": "ML & AI", "computer_vision": "ML & AI", "mlops": "ML & AI",
    "llm": "ML & AI", "langchain": "ML & AI", "openai": "ML & AI",
    "sagemaker": "ML & AI",
    "spark": "Data Engineering", "pyspark": "Data Engineering",
    "airflow": "Data Engineering", "kafka": "Data Engineering",
    "dbt": "Data Engineering", "hadoop": "Data Engineering",
    "databricks": "Data Engineering",
    "git": "DevOps & Tools", "linux": "DevOps & Tools",
    "ci_cd": "DevOps & Tools", "jenkins": "DevOps & Tools",
    "github_actions": "DevOps & Tools", "gitlab": "DevOps & Tools",
}

COUNTRIES = ["US", "GB", "DE", "CA", "AU", "IN", "FR", "NL", "SG", "IE"]


def _get_category(skill: str) -> str:
    return SKILL_CATEGORIES.get(skill, "Other")


# ── demo data generators ────────────────────────────────────────────────────


def _generate_demo_job_skills(n_weeks: int = 24) -> pd.DataFrame:
    """Generate realistic weekly skill demand data."""
    rng = np.random.default_rng(42)

    top_skills = list(SKILL_CATEGORIES.keys())
    base_demand = {s: rng.integers(5, 300) for s in top_skills}
    base_salary = {s: rng.integers(60_000, 200_000) for s in top_skills}

    rows = []
    end_date = datetime.now()
    for w in range(n_weeks):
        week_start = end_date - timedelta(weeks=n_weeks - 1 - w)
        for skill in top_skills:
            for country in rng.choice(COUNTRIES, size=rng.integers(2, 6), replace=False):
                trend = 1 + 0.02 * w * (1 if rng.random() > 0.3 else -0.5)
                noise = rng.normal(1, 0.15)
                job_count = max(1, int(base_demand[skill] * trend * noise * 0.3))
                salary = max(
                    30_000,
                    int(base_salary[skill] * rng.normal(1, 0.08)),
                )
                rows.append({
                    "skill": skill,
                    "week": week_start.strftime("%Y-%m-%d"),
                    "country": country,
                    "job_count": job_count,
                    "avg_salary": salary,
                    "job_id": f"job_{skill}_{country}_{w}",
                })

    df = pd.DataFrame(rows)
    df["week"] = pd.to_datetime(df["week"])
    df["posted_date"] = df["week"]
    df["salary_mid_usd"] = df["avg_salary"]
    return df


def _demo_top_skills(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["skill", "country"])
        .agg(job_count=("job_count", "sum"),
             unique_jobs=("job_id", "nunique"),
             avg_salary=("avg_salary", "mean"))
        .reset_index()
        .round({"avg_salary": 0})
    )


def _demo_skill_trends(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["skill", "week", "country"])
        .agg(job_count=("job_count", "sum"), avg_salary=("avg_salary", "mean"))
        .reset_index()
    )


def _demo_salary_by_country(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("country")
        .agg(avg_salary=("avg_salary", "mean"),
             job_count=("job_count", "sum"),
             unique_skills=("skill", "nunique"))
        .reset_index()
        .round({"avg_salary": 0})
    )


def _demo_skill_growth(df: pd.DataFrame) -> pd.DataFrame:
    weekly = (
        df.groupby(["skill", "week"])
        .agg(job_count=("job_count", "sum"), avg_salary=("avg_salary", "mean"))
        .reset_index()
        .sort_values(["skill", "week"])
    )

    results = []
    for skill, grp in weekly.groupby("skill"):
        if len(grp) < 8:
            continue
        recent = grp.tail(4)["job_count"].sum()
        prior = grp.iloc[-8:-4]["job_count"].sum()
        growth_pct = ((recent - prior) / max(prior, 1)) * 100

        total_jobs = int(grp["job_count"].sum())
        avg_weekly = round(grp["job_count"].mean(), 1)
        max_salary = int(grp["avg_salary"].max())
        forecast = round(avg_weekly * (1 + growth_pct / 100), 1)

        if growth_pct > 50:
            trend = "Hot"
        elif growth_pct > 20:
            trend = "Rising"
        elif growth_pct > 0:
            trend = "Growing"
        else:
            trend = "Declining"

        results.append({
            "skill": skill, "total_jobs": total_jobs,
            "avg_weekly_jobs": avg_weekly, "max_salary": max_salary,
            "growth_pct": round(growth_pct, 1),
            "forecast_weekly": forecast, "trend_status": trend,
        })

    return pd.DataFrame(results)


def _demo_emerging_skills(df: pd.DataFrame) -> pd.DataFrame:
    max_week = df["week"].max()
    cutoff = max_week - timedelta(weeks=4)
    recent = (
        df[df["week"] >= cutoff]
        .groupby("skill")
        .agg(current_jobs=("job_count", "sum"), avg_salary=("avg_salary", "mean"))
        .reset_index()
    )
    prior = (
        df[(df["week"] >= cutoff - timedelta(weeks=4)) & (df["week"] < cutoff)]
        .groupby("skill")
        .agg(prev_jobs=("job_count", "sum"))
        .reset_index()
    )

    merged = recent.merge(prior, on="skill", how="left").fillna({"prev_jobs": 1})
    merged["growth_pct"] = ((merged["current_jobs"] - merged["prev_jobs"]) / merged["prev_jobs"] * 100).round(1)
    merged["avg_salary"] = merged["avg_salary"].round(0)

    merged["trend_status"] = "Stable"
    merged.loc[(merged["growth_pct"] > 10), "trend_status"] = "Growing"
    merged.loc[(merged["growth_pct"] > 25) & (merged["current_jobs"] >= 3), "trend_status"] = "Rising"
    merged.loc[(merged["growth_pct"] > 50) & (merged["current_jobs"] >= 5), "trend_status"] = "Hot"

    return merged[merged["current_jobs"] >= 3].sort_values("growth_pct", ascending=False)


def _demo_cooccurrence(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    skill_list = df["skill"].unique()
    pairs = []

    common_pairs = [
        ("python", "aws"), ("python", "docker"), ("python", "sql"),
        ("javascript", "react"), ("javascript", "typescript"), ("javascript", "nodejs"),
        ("aws", "docker"), ("aws", "kubernetes"), ("aws", "terraform"),
        ("docker", "kubernetes"), ("java", "spring"), ("python", "machine_learning"),
        ("python", "pytorch"), ("python", "tensorflow"), ("react", "typescript"),
        ("kubernetes", "docker"), ("spark", "python"), ("airflow", "python"),
        ("postgresql", "python"), ("mongodb", "nodejs"), ("kafka", "java"),
        ("gcp", "kubernetes"), ("azure", "docker"), ("ci_cd", "docker"),
        ("linux", "docker"), ("git", "ci_cd"), ("redis", "nodejs"),
        ("flask", "python"), ("django", "python"), ("fastapi", "python"),
    ]

    for a, b in common_pairs:
        if a in skill_list and b in skill_list:
            sa, sb = sorted([a, b])
            pairs.append({
                "skill_a": sa, "skill_b": sb,
                "cooccurrence_count": int(rng.integers(10, 200)),
            })

    for _ in range(40):
        a, b = rng.choice(skill_list, size=2, replace=False)
        sa, sb = sorted([a, b])
        pairs.append({
            "skill_a": sa, "skill_b": sb,
            "cooccurrence_count": int(rng.integers(3, 50)),
        })

    result = pd.DataFrame(pairs).drop_duplicates(subset=["skill_a", "skill_b"])
    return result[result["cooccurrence_count"] >= 3]


def _demo_kpis(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{
        "total_jobs": int(df["job_id"].nunique()),
        "unique_skills": int(df["skill"].nunique()),
        "avg_salary": int(df["avg_salary"].mean()),
        "countries": int(df["country"].nunique()),
    }])


def _demo_skills_by_category(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("skill").agg(job_count=("job_count", "sum")).reset_index()
    agg["category"] = agg["skill"].apply(_get_category)
    return agg


# ── public loader class ─────────────────────────────────────────────────────


class DashboardDataLoader:
    """
    Loads analytics data for the dashboard.
    Auto-falls-back to demo data when AWS is not configured.
    """

    def __init__(
        self,
        database: str = "job_market_db",
        region: str = "us-east-1",
        use_demo: Optional[bool] = None,
    ):
        self.database = database
        self.region = region

        if use_demo is None:
            self.use_demo = not (_HAS_AWS and self._aws_configured())
        else:
            self.use_demo = use_demo

        self._demo_raw: Optional[pd.DataFrame] = None
        self._boto3_session = (
            boto3.Session(region_name=self.region) if _HAS_AWS else None
        )

    @staticmethod
    def _aws_configured() -> bool:
        try:
            sts = boto3.client("sts")
            sts.get_caller_identity()
            return True
        except Exception:
            return False

    def _get_demo_raw(self) -> pd.DataFrame:
        if self._demo_raw is None:
            self._demo_raw = _generate_demo_job_skills()
        return self._demo_raw

    def _query_athena(self, view_name: str) -> pd.DataFrame:
        try:
            return wr.athena.read_sql_query(
                f"SELECT * FROM {self.database}.{view_name}",
                database=self.database,
                boto3_session=self._boto3_session,
            )
        except Exception:
            self.use_demo = True
            raise

    def _load(self, view_name: str, demo_fn):
        """Try Athena; on any failure, fall back to demo data."""
        if not self.use_demo:
            try:
                return self._query_athena(view_name)
            except Exception:
                self.use_demo = True
        return demo_fn(self._get_demo_raw())

    def load_kpis(self) -> pd.DataFrame:
        return self._load("vw_dashboard_kpis", _demo_kpis)

    def load_top_skills(self) -> pd.DataFrame:
        return self._load("vw_top_skills", _demo_top_skills)

    def load_skill_trends(self) -> pd.DataFrame:
        return self._load("vw_skill_trends", _demo_skill_trends)

    def load_salary_by_country(self) -> pd.DataFrame:
        return self._load("vw_salary_by_country", _demo_salary_by_country)

    def load_skill_growth(self) -> pd.DataFrame:
        return self._load("vw_skill_growth", _demo_skill_growth)

    def load_emerging_skills(self) -> pd.DataFrame:
        return self._load("vw_emerging_skills", _demo_emerging_skills)

    def load_cooccurrence(self) -> pd.DataFrame:
        return self._load("vw_skill_cooccurrence", _demo_cooccurrence)

    def load_skills_by_category(self) -> pd.DataFrame:
        return self._load("vw_skills_by_category", _demo_skills_by_category)

    def load_model_metrics(self) -> Dict:
        metrics_path = os.path.join(
            os.path.dirname(__file__), "..", "ml", "models", "metrics.json"
        )
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                return json.load(f)
        return {}

    def is_demo_mode(self) -> bool:
        return self.use_demo
