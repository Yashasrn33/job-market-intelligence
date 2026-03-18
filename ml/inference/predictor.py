"""
Skill Demand Predictor - Inference Module

Provides production-ready prediction APIs:
- forecast_skill_demand()      Predict future demand for skills
- detect_emerging_skills()     Find breakout skills
- get_skill_recommendations()  "Learn X based on Y" suggestions
- generate_market_report()     Full market intelligence report

Works with either a local model directory or a live SageMaker endpoint.
"""

import json
import logging
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional

import awswrangler as wr
import boto3
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SkillDemandPredictor:
    """Production inference class for skill demand predictions."""

    EMERGENCE_FEATURES = [
        "mom_growth",
        "qoq_growth",
        "growth_acceleration",
        "market_share_change",
        "trend_strength_8w",
        "emergence_score",
    ]

    def __init__(
        self,
        model_path: Optional[str] = None,
        sagemaker_endpoint: Optional[str] = None,
        database: str = "job_market_db",
        region: str = "us-east-1",
    ):
        self.database = database
        self.region = region
        self.sagemaker_endpoint = sagemaker_endpoint
        self.boto3_session = boto3.Session(region_name=self.region)

        if model_path:
            self._load_local_model(model_path)
        elif sagemaker_endpoint:
            self.runtime = boto3.client("sagemaker-runtime", region_name=region)
        else:
            raise ValueError("Provide either model_path or sagemaker_endpoint")

    # ── model loading ───────────────────────────────────────────────────────

    def _load_local_model(self, model_path: str) -> None:
        blobs = {
            "demand_model": "demand_model.pkl",
            "emergence_model": "emergence_model.pkl",
            "cluster_model": "cluster_model.pkl",
            "scaler": "scaler.pkl",
        }
        for attr, fname in blobs.items():
            with open(os.path.join(model_path, fname), "rb") as fh:
                setattr(self, attr, pickle.load(fh))

        with open(os.path.join(model_path, "feature_cols.json")) as fh:
            self.feature_cols: List[str] = json.load(fh)

        metrics_path = os.path.join(model_path, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as fh:
                self.metrics: Dict[str, Any] = json.load(fh)
        else:
            self.metrics = {}

        logger.info("Loaded model from %s (%d features)", model_path, len(self.feature_cols))

    # ── Athena helpers ──────────────────────────────────────────────────────

    def _get_latest_features(self, skills: Optional[List[str]] = None) -> pd.DataFrame:
        skill_filter = ""
        if skills:
            skill_list = ", ".join(f"'{s}'" for s in skills)
            skill_filter = f"AND skill IN ({skill_list})"

        query = f"""
        WITH latest_week AS (
            SELECT MAX(DATE_TRUNC('week', posted_date)) AS max_week
            FROM {self.database}.job_skills
        ),
        weekly_stats AS (
            SELECT
                skill,
                DATE_TRUNC('week', posted_date) AS week,
                COUNT(*)            AS job_count,
                AVG(salary_mid_usd) AS avg_salary
            FROM {self.database}.job_skills
            WHERE posted_date >= DATE_ADD('week', -16,
                                          (SELECT max_week FROM latest_week))
                  {skill_filter}
            GROUP BY skill, DATE_TRUNC('week', posted_date)
        )
        SELECT
            w1.skill,
            w1.week,
            w1.job_count,
            w1.avg_salary,
            LAG(w1.job_count, 1) OVER (PARTITION BY w1.skill ORDER BY w1.week) AS job_count_lag_1w,
            LAG(w1.job_count, 4) OVER (PARTITION BY w1.skill ORDER BY w1.week) AS job_count_lag_4w,
            LAG(w1.avg_salary, 1) OVER (PARTITION BY w1.skill ORDER BY w1.week) AS salary_lag_1w,
            AVG(w1.job_count) OVER (
                PARTITION BY w1.skill ORDER BY w1.week
                ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
            ) AS job_count_ma_4w,
            AVG(w1.job_count) OVER (
                PARTITION BY w1.skill ORDER BY w1.week
                ROWS BETWEEN 7 PRECEDING AND CURRENT ROW
            ) AS job_count_ma_8w
        FROM weekly_stats w1
        WHERE w1.week = (SELECT max_week FROM latest_week)
        ORDER BY w1.job_count DESC
        """

        df = wr.athena.read_sql_query(
            query, database=self.database, boto3_session=self.boto3_session
        )

        df["wow_growth"] = (
            (df["job_count"] - df["job_count_lag_1w"])
            / (df["job_count_lag_1w"] + 1)
        )
        df["mom_growth"] = (
            (df["job_count"] - df["job_count_lag_4w"])
            / (df["job_count_lag_4w"] + 1)
        )

        return df

    def _get_weekly_history(
        self, skills: Optional[List[str]] = None, lookback_weeks: int = 16
    ) -> pd.DataFrame:
        """Fetch weekly aggregates for the last *lookback_weeks* for feature engineering."""
        skill_filter = ""
        if skills:
            skill_list = ", ".join(f"'{s}'" for s in skills)
            skill_filter = f"AND skill IN ({skill_list})"

        query = f"""
        WITH latest_week AS (
            SELECT MAX(DATE_TRUNC('week', posted_date)) AS max_week
            FROM {self.database}.job_skills
        )
        SELECT
            skill,
            DATE_TRUNC('week', posted_date) AS week,
            COUNT(*)            AS job_count,
            AVG(salary_mid_usd) AS avg_salary
        FROM {self.database}.job_skills
        WHERE posted_date >= DATE_ADD('week', -{lookback_weeks},
                                      (SELECT max_week FROM latest_week))
              {skill_filter}
        GROUP BY skill, DATE_TRUNC('week', posted_date)
        ORDER BY skill, week
        """

        return wr.athena.read_sql_query(
            query, database=self.database, boto3_session=self.boto3_session
        )

    def _align_features(self, df: pd.DataFrame) -> np.ndarray:
        """Ensure df has every expected feature column, then scale."""
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        X = df[self.feature_cols].fillna(0).values
        return self.scaler.transform(X)

    # ── public APIs ─────────────────────────────────────────────────────────

    def forecast_skill_demand(
        self,
        skills: Optional[List[str]] = None,
        horizon_weeks: int = 4,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """Forecast demand for skills over the next *horizon_weeks*."""

        logger.info("Forecasting demand for %d weeks …", horizon_weeks)

        df = self._get_latest_features(skills)
        if skills is None:
            df = df.head(top_n)

        X_scaled = self._align_features(df)
        predictions = self.demand_model.predict(X_scaled)

        results = pd.DataFrame(
            {
                "skill": df["skill"].values,
                "current_demand": df["job_count"].values,
                "predicted_demand": np.round(predictions).astype(int),
                "current_salary": df["avg_salary"].values,
            }
        )

        results["demand_change"] = (
            results["predicted_demand"] - results["current_demand"]
        )
        results["growth_pct"] = (
            ((results["predicted_demand"] / results["current_demand"]) - 1) * 100
        ).round(1)

        return results.sort_values("growth_pct", ascending=False)

    def detect_emerging_skills(
        self, threshold: float = 0.8, top_n: int = 10
    ) -> pd.DataFrame:
        """Detect skills showing unusual growth patterns."""

        logger.info("Detecting emerging skills …")

        hist = self._get_weekly_history(lookback_weeks=20)
        hist = hist.sort_values(["skill", "week"])

        # Growth features (match training feature engineering)
        hist["wow_growth"] = hist.groupby("skill")["job_count"].pct_change(1)
        hist["mom_growth"] = hist.groupby("skill")["job_count"].pct_change(4)
        hist["qoq_growth"] = hist.groupby("skill")["job_count"].pct_change(12)
        hist["growth_acceleration"] = hist.groupby("skill")["wow_growth"].diff()

        overall_weekly = hist.groupby("week")["job_count"].transform("sum")
        hist["market_share"] = hist["job_count"] / (overall_weekly + 1e-8)
        hist["market_share_change"] = hist.groupby("skill")["market_share"].pct_change(4)

        def _trend_strength(x: np.ndarray) -> float:
            if len(x) < 4:
                return np.nan
            t = np.arange(len(x))
            try:
                slope, _ = np.polyfit(t, x, 1)
                predicted = slope * t + np.mean(x) - slope * np.mean(t)
                ss_res = np.sum((x - predicted) ** 2)
                ss_tot = np.sum((x - np.mean(x)) ** 2)
                return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            except Exception:
                return np.nan

        hist["trend_strength_8w"] = hist.groupby("skill")["job_count"].transform(
            lambda x: x.rolling(8, min_periods=4).apply(
                lambda arr: _trend_strength(arr), raw=True
            )
        )

        # Composite emergence_score (same weights as training)
        def _norm_weekly(col: str) -> pd.Series:
            return hist.groupby("week")[col].transform(
                lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
            )

        hist["growth_score"] = _norm_weekly("mom_growth").clip(0, 1)
        hist["acceleration_score"] = _norm_weekly("growth_acceleration").clip(0, 1)
        hist["trend_score"] = hist["trend_strength_8w"].fillna(0).clip(0, 1)

        historical_avg = hist.groupby("skill")["job_count"].transform("mean")
        hist["novelty_score"] = 1 / (1 + np.log1p(historical_avg))
        hist["novelty_score"] = _norm_weekly("novelty_score")

        hist["emergence_score"] = (
            0.35 * hist["growth_score"]
            + 0.25 * hist["acceleration_score"]
            + 0.25 * hist["trend_score"]
            + 0.15 * hist["novelty_score"]
        )

        # Score only the latest row per skill.
        latest = hist.sort_values("week").groupby("skill").tail(1).copy()

        X_scaled = self._align_features(latest)
        eidx = [self.feature_cols.index(f) for f in self.EMERGENCE_FEATURES if f in self.feature_cols]
        X_em = X_scaled[:, eidx]

        raw_scores = -self.emergence_model.decision_function(X_em)
        score_min, score_max = raw_scores.min(), raw_scores.max()
        latest["emergence_score_model"] = (raw_scores - score_min) / (score_max - score_min + 1e-8)

        emerging = latest[
            (latest["emergence_score_model"] >= threshold) & (latest["mom_growth"] > 0.1)
        ].copy()

        emerging = emerging.sort_values("emergence_score_model", ascending=False).head(top_n)

        out = emerging[
            ["skill", "job_count", "emergence_score_model", "mom_growth", "avg_salary"]
        ].copy()
        out.columns = [
            "skill",
            "current_jobs",
            "emergence_score",
            "monthly_growth_pct",
            "avg_salary",
        ]
        out["monthly_growth_pct"] = (out["monthly_growth_pct"] * 100).round(1)
        out["emergence_score"] = out["emergence_score"].round(3)

        return out

    def get_skill_recommendations(
        self, known_skills: List[str], top_n: int = 5
    ) -> Dict[str, Any]:
        """'If you know X, you should also learn Y' via cluster co-membership."""

        logger.info("Getting recommendations for %s …", known_skills)

        cluster_mapping: Dict[str, int] = (
            self.metrics.get("cluster_model", {}).get("skill_cluster_mapping", {})
        )
        if not cluster_mapping:
            return {"error": "Cluster model not available"}

        known_clusters = {
            cluster_mapping[s.lower()]
            for s in known_skills
            if s.lower() in cluster_mapping
        }

        recommendations: List[Dict[str, Any]] = []
        for skill, cluster in cluster_mapping.items():
            if cluster in known_clusters and skill.lower() not in {
                s.lower() for s in known_skills
            }:
                recommendations.append(
                    {
                        "skill": skill,
                        "cluster": cluster,
                        "reason": f"Often used alongside {known_skills[0]}",
                    }
                )

        if recommendations:
            rec_skills = [r["skill"] for r in recommendations[: top_n * 2]]
            try:
                forecast = self.forecast_skill_demand(skills=rec_skills, top_n=top_n * 2)
                for rec in recommendations:
                    row = forecast[forecast["skill"] == rec["skill"]]
                    if not row.empty:
                        rec["current_demand"] = int(row["current_demand"].iloc[0])
                        rec["growth_pct"] = float(row["growth_pct"].iloc[0])
                        rec["salary"] = float(row["current_salary"].iloc[0])
            except Exception as exc:
                logger.warning("Could not enrich recommendations: %s", exc)

        recommendations.sort(key=lambda r: r.get("growth_pct", 0), reverse=True)

        return {
            "known_skills": known_skills,
            "recommendations": recommendations[:top_n],
            "cluster_ids": list(known_clusters),
        }

    def generate_market_report(self) -> Dict[str, Any]:
        """Generate a comprehensive market intelligence report."""

        logger.info("Generating market intelligence report …")

        all_forecasts = self.forecast_skill_demand(top_n=50)
        emerging = self.detect_emerging_skills(threshold=0.7, top_n=10)

        top_demand = (
            all_forecasts.nlargest(10, "current_demand")[
                ["skill", "current_demand", "predicted_demand", "growth_pct"]
            ]
            .to_dict("records")
        )

        fastest_growing = (
            all_forecasts[all_forecasts["current_demand"] >= 10]
            .nlargest(10, "growth_pct")[
                ["skill", "current_demand", "predicted_demand", "growth_pct"]
            ]
            .to_dict("records")
        )

        highest_paying = (
            all_forecasts.nlargest(10, "current_salary")[
                ["skill", "current_salary", "current_demand", "growth_pct"]
            ]
            .to_dict("records")
        )

        summary = {
            "total_skills_tracked": len(all_forecasts),
            "avg_growth_rate": float(all_forecasts["growth_pct"].mean()),
            "skills_growing": int((all_forecasts["growth_pct"] > 0).sum()),
            "skills_declining": int((all_forecasts["growth_pct"] < 0).sum()),
            "avg_salary": float(all_forecasts["current_salary"].mean()),
            "report_date": datetime.now().isoformat(),
        }

        return {
            "market_summary": summary,
            "top_demand_skills": top_demand,
            "fastest_growing_skills": fastest_growing,
            "emerging_skills": emerging.to_dict("records"),
            "highest_paying_skills": highest_paying,
        }


# ── convenience factory ─────────────────────────────────────────────────────


def get_predictor(
    model_path: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> SkillDemandPredictor:
    return SkillDemandPredictor(model_path=model_path, sagemaker_endpoint=endpoint)


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Skill demand predictions")
    parser.add_argument("--model-path", required=True, help="Path to model dir")
    parser.add_argument(
        "--action",
        choices=["forecast", "emerging", "recommend", "report"],
        default="report",
    )
    parser.add_argument("--skills", nargs="+", help="Skills for forecast/recommend")
    args = parser.parse_args()

    predictor = SkillDemandPredictor(model_path=args.model_path)

    if args.action == "forecast":
        res = predictor.forecast_skill_demand(skills=args.skills)
        print("\nSKILL DEMAND FORECAST")
        print(res.to_string(index=False))

    elif args.action == "emerging":
        res = predictor.detect_emerging_skills()
        print("\nEMERGING SKILLS")
        print(res.to_string(index=False))

    elif args.action == "recommend":
        if not args.skills:
            print("Error: --skills required for recommendations")
        else:
            res = predictor.get_skill_recommendations(args.skills)
            print(f"\nRECOMMENDATIONS FOR: {', '.join(args.skills)}")
            for r in res["recommendations"]:
                print(
                    f"  - {r['skill']}: {r.get('growth_pct', 'N/A')}% growth, "
                    f"${r.get('salary', 0):,.0f} avg salary"
                )

    elif args.action == "report":
        report = predictor.generate_market_report()
        print("\n" + "=" * 60)
        print("JOB MARKET INTELLIGENCE REPORT")
        print("=" * 60)

        print("\nMarket Summary:")
        for k, v in report["market_summary"].items():
            print(f"  {k}: {v}")

        print("\nTop Demand Skills:")
        for s in report["top_demand_skills"][:5]:
            print(f"  - {s['skill']}: {s['current_demand']} jobs ({s['growth_pct']:+.1f}%)")

        print("\nFastest Growing:")
        for s in report["fastest_growing_skills"][:5]:
            print(f"  - {s['skill']}: {s['growth_pct']:+.1f}% growth")

        print("\nEmerging Skills:")
        for s in report["emerging_skills"][:5]:
            print(f"  - {s['skill']}: emergence score {s['emergence_score']:.2f}")

        print("\nHighest Paying:")
        for s in report["highest_paying_skills"][:5]:
            print(f"  - {s['skill']}: ${s['current_salary']:,.0f}")
