"""
Feature Engineering for Skill Demand Forecasting
Transforms raw job_skills data into ML-ready features.

Features created:
- Lag features (1, 2, 4, 8, 12 weeks)
- Rolling averages (4, 8, 12 weeks)
- Growth rates (week-over-week, month-over-month, quarter-over-quarter)
- Seasonality indicators (cyclical encoding)
- Skill co-occurrence / category features
- Salary trend features
- Volatility & trend-strength metrics
- Composite emergence score for breakout detection
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import awswrangler as wr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SkillFeatureEngineer:
    """Creates ML features from job_skills data for demand forecasting."""

    def __init__(self, database: str = "job_market_db", region: str = "us-east-1"):
        self.database = database
        self.region = region

    # ── data loading ────────────────────────────────────────────────────────

    def load_skill_data(self, lookback_weeks: int = 52) -> pd.DataFrame:
        """Load historical weekly skill demand from Athena."""

        query = f"""
        SELECT
            skill,
            DATE_TRUNC('week', posted_date) AS week,
            country,
            COUNT(*)                        AS job_count,
            AVG(salary_mid_usd)             AS avg_salary,
            COUNT(DISTINCT job_id)          AS unique_jobs
        FROM {self.database}.job_skills
        WHERE posted_date >= DATE_ADD('week', -{lookback_weeks}, CURRENT_DATE)
        GROUP BY skill, DATE_TRUNC('week', posted_date), country
        ORDER BY skill, week
        """

        logger.info("Loading %d weeks of skill data…", lookback_weeks)
        df = wr.athena.read_sql_query(
            query, database=self.database, region_name=self.region
        )
        logger.info(
            "Loaded %d records for %d skills", len(df), df["skill"].nunique()
        )
        return df

    def load_skill_data_from_s3(
        self,
        bucket: str,
        data_source: str = "combined",
    ) -> pd.DataFrame:
        """
        Load raw job_skills data from S3 parquet and aggregate to weekly level.

        Tries Athena first, falls back to direct S3 parquet reads.

        Args:
            bucket: S3 bucket name.
            data_source: 'kaggle', 'adzuna', or 'combined'.
        """
        paths = {
            "kaggle": [f"s3://{bucket}/processed/job_skills_kaggle/"],
            "adzuna": [f"s3://{bucket}/processed/job_skills/"],
            "combined": [
                f"s3://{bucket}/processed/job_skills/",
                f"s3://{bucket}/processed/job_skills_kaggle/",
            ],
        }

        if data_source not in paths:
            raise ValueError(
                f"Unknown data_source '{data_source}'. "
                f"Choose from: {list(paths.keys())}"
            )

        frames: list[pd.DataFrame] = []
        for path in paths[data_source]:
            try:
                df = wr.s3.read_parquet(path)
                frames.append(df)
                logger.info("Loaded %d records from %s", len(df), path)
            except Exception as exc:
                logger.warning("Could not read %s: %s", path, exc)

        if not frames:
            raise RuntimeError(
                f"No data found for source '{data_source}' in bucket '{bucket}'"
            )

        raw = pd.concat(frames, ignore_index=True)
        return self._aggregate_to_weekly(raw)

    @staticmethod
    def _aggregate_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate raw job_skills rows to weekly skill counts."""
        df = df.copy()
        df["posted_date"] = pd.to_datetime(df["posted_date"], errors="coerce")
        df = df.dropna(subset=["posted_date"])

        df["week"] = df["posted_date"].dt.to_period("W").dt.start_time

        salary_col = (
            "salary_mid_usd" if "salary_mid_usd" in df.columns
            else "salary_mid" if "salary_mid" in df.columns
            else None
        )

        agg_dict: dict = {}
        if "job_id" in df.columns:
            agg_dict["job_id"] = "count"
        else:
            agg_dict["skill"] = "count"

        if salary_col:
            agg_dict[salary_col] = "mean"

        grouped = df.groupby(["skill", "week", "country"]).agg(agg_dict).reset_index()

        count_col = "job_id" if "job_id" in agg_dict else "skill"
        grouped = grouped.rename(columns={count_col: "job_count"})

        if salary_col and salary_col != "avg_salary":
            grouped = grouped.rename(columns={salary_col: "avg_salary"})

        if "avg_salary" not in grouped.columns:
            grouped["avg_salary"] = np.nan

        return grouped

    # ── feature blocks ──────────────────────────────────────────────────────

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["week"] = pd.to_datetime(df["week"])

        df["week_of_year"] = df["week"].dt.isocalendar().week.astype(int)
        df["month"] = df["week"].dt.month
        df["quarter"] = df["week"].dt.quarter
        df["year"] = df["week"].dt.year

        df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
        df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        return df

    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["skill", "week"])

        for lag in [1, 2, 4, 8, 12]:
            df[f"job_count_lag_{lag}w"] = (
                df.groupby("skill")["job_count"].shift(lag)
            )
            df[f"salary_lag_{lag}w"] = (
                df.groupby("skill")["avg_salary"].shift(lag)
            )

        return df

    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["skill", "week"])

        for window in [4, 8, 12]:
            df[f"job_count_ma_{window}w"] = df.groupby("skill")[
                "job_count"
            ].transform(lambda x: x.rolling(window, min_periods=1).mean())

            df[f"job_count_std_{window}w"] = df.groupby("skill")[
                "job_count"
            ].transform(lambda x: x.rolling(window, min_periods=2).std())

            df[f"salary_ma_{window}w"] = df.groupby("skill")[
                "avg_salary"
            ].transform(lambda x: x.rolling(window, min_periods=1).mean())

        df["job_count_ema_4w"] = df.groupby("skill")["job_count"].transform(
            lambda x: x.ewm(span=4, adjust=False).mean()
        )

        return df

    def create_growth_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["skill", "week"])

        df["wow_growth"] = df.groupby("skill")["job_count"].pct_change(1)
        df["mom_growth"] = df.groupby("skill")["job_count"].pct_change(4)
        df["qoq_growth"] = df.groupby("skill")["job_count"].pct_change(12)

        df["growth_acceleration"] = df.groupby("skill")["wow_growth"].diff()

        overall_weekly = df.groupby("week")["job_count"].transform("sum")
        df["market_share"] = df["job_count"] / overall_weekly
        df["market_share_change"] = (
            df.groupby("skill")["market_share"].pct_change(4)
        )

        df["salary_wow_growth"] = (
            df.groupby("skill")["avg_salary"].pct_change(1)
        )
        df["salary_mom_growth"] = (
            df.groupby("skill")["avg_salary"].pct_change(4)
        )

        return df

    def create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["skill", "week"])

        df["cv_8w"] = df["job_count_std_8w"] / (df["job_count_ma_8w"] + 1)

        def _trend_strength(x):
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

        df["trend_strength_8w"] = df.groupby("skill")["job_count"].transform(
            lambda x: x.rolling(8, min_periods=4).apply(
                _trend_strength, raw=True
            )
        )

        return df

    def create_emergence_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Composite emergence score combining:
        - Rapid growth (35 %)
        - Acceleration   (25 %)
        - Trend strength (25 %)
        - Novelty        (15 %)
        """
        df = df.copy()

        def _norm_weekly(col: str) -> pd.Series:
            return df.groupby("week")[col].transform(
                lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
            )

        df["growth_score"] = _norm_weekly("mom_growth").clip(0, 1)
        df["acceleration_score"] = _norm_weekly("growth_acceleration").clip(0, 1)
        df["trend_score"] = df["trend_strength_8w"].fillna(0).clip(0, 1)

        historical_avg = df.groupby("skill")["job_count"].transform("mean")
        df["novelty_score"] = 1 / (1 + np.log1p(historical_avg))
        df["novelty_score"] = _norm_weekly("novelty_score")

        df["emergence_score"] = (
            0.35 * df["growth_score"]
            + 0.25 * df["acceleration_score"]
            + 0.25 * df["trend_score"]
            + 0.15 * df["novelty_score"]
        )

        return df

    def create_skill_cooccurrence_features(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add skill-category aggregates and within-category share."""

        skill_categories = {
            "python": "languages",
            "java": "languages",
            "javascript": "languages",
            "react": "frontend",
            "angular": "frontend",
            "vue": "frontend",
            "nodejs": "backend",
            "django": "backend",
            "flask": "backend",
            "aws": "cloud",
            "azure": "cloud",
            "gcp": "cloud",
            "kubernetes": "cloud",
            "postgresql": "databases",
            "mongodb": "databases",
            "redis": "databases",
            "machine learning": "ml",
            "deep learning": "ml",
            "pytorch": "ml",
            "tensorflow": "ml",
            "spark": "data_eng",
            "airflow": "data_eng",
            "kafka": "data_eng",
        }

        df["skill_category"] = df["skill"].map(skill_categories).fillna("other")

        cat_weekly = (
            df.groupby(["week", "skill_category"])["job_count"]
            .sum()
            .reset_index()
            .rename(columns={"job_count": "category_job_count"})
        )

        df = df.merge(cat_weekly, on=["week", "skill_category"], how="left")
        df["category_share"] = df["job_count"] / (df["category_job_count"] + 1)

        return df

    # ── pipeline ────────────────────────────────────────────────────────────

    FEATURE_COLS: List[str] = [
        # Lags
        "job_count_lag_1w",
        "job_count_lag_2w",
        "job_count_lag_4w",
        "salary_lag_1w",
        "salary_lag_4w",
        # Rolling
        "job_count_ma_4w",
        "job_count_ma_8w",
        "job_count_ma_12w",
        "job_count_std_4w",
        "job_count_std_8w",
        "job_count_ema_4w",
        "salary_ma_4w",
        "salary_ma_8w",
        # Growth
        "wow_growth",
        "mom_growth",
        "qoq_growth",
        "growth_acceleration",
        "market_share",
        "market_share_change",
        "salary_wow_growth",
        "salary_mom_growth",
        # Volatility
        "cv_8w",
        "trend_strength_8w",
        # Emergence
        "emergence_score",
        "growth_score",
        "trend_score",
        # Category
        "category_share",
        "category_job_count",
        # Seasonality
        "week_sin",
        "week_cos",
        "month_sin",
        "month_cos",
    ]

    def prepare_training_data(
        self,
        target_horizon: int = 4,
        raw_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Full pipeline: load → engineer features → create target.

        Args:
            target_horizon: weeks ahead to predict.
            raw_df: optional pre-loaded weekly aggregated data. When provided
                the Athena load step is skipped.

        Returns:
            (features_df, target_df) ready for model training.
        """

        logger.info("Starting feature engineering pipeline…")

        if raw_df is not None:
            df = raw_df.copy()
            logger.info(
                "Using provided DataFrame: %d records, %d skills",
                len(df),
                df["skill"].nunique(),
            )
        else:
            df = self.load_skill_data(lookback_weeks=52)

        logger.info("Creating time features…")
        df = self.create_time_features(df)

        logger.info("Creating lag features…")
        df = self.create_lag_features(df)

        logger.info("Creating rolling features…")
        df = self.create_rolling_features(df)

        logger.info("Creating growth features…")
        df = self.create_growth_features(df)

        logger.info("Creating volatility features…")
        df = self.create_volatility_features(df)

        logger.info("Creating emergence score…")
        df = self.create_emergence_score(df)

        logger.info("Creating co-occurrence features…")
        df = self.create_skill_cooccurrence_features(df)

        # Target: job_count shifted forward by target_horizon weeks
        df["target"] = df.groupby("skill")["job_count"].shift(-target_horizon)

        df = df.dropna(subset=["target"])

        df[self.FEATURE_COLS] = (
            df[self.FEATURE_COLS]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )

        features_df = df[["skill", "week", "country"] + self.FEATURE_COLS]
        target_df = df[["skill", "week", "target"]]

        logger.info(
            "Prepared %d samples with %d features",
            len(features_df),
            len(self.FEATURE_COLS),
        )

        return features_df, target_df

    def prepare_training_data_from_s3(
        self,
        bucket: str,
        data_source: str = "combined",
        target_horizon: int = 4,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        End-to-end: load raw job_skills from S3, engineer features, create target.

        Args:
            bucket: S3 bucket name.
            data_source: 'kaggle', 'adzuna', or 'combined'.
            target_horizon: weeks ahead to predict.

        Returns:
            (features_df, target_df) ready for model training.
        """
        raw_df = self.load_skill_data_from_s3(bucket, data_source)
        return self.prepare_training_data(
            target_horizon=target_horizon, raw_df=raw_df
        )

    def save_features(
        self,
        bucket: str,
        prefix: str = "ml/features",
        data_source: Optional[str] = None,
    ) -> str:
        """Save engineered features to S3 as Parquet."""

        if data_source:
            features_df, target_df = self.prepare_training_data_from_s3(
                bucket, data_source
            )
        else:
            features_df, target_df = self.prepare_training_data()

        training_data = features_df.merge(target_df, on=["skill", "week"])

        output_path = f"s3://{bucket}/{prefix}/training_data.parquet"
        wr.s3.to_parquet(training_data, output_path)

        logger.info("Saved training data to %s", output_path)
        return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate ML features")
    parser.add_argument("--database", default="job_market_db")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--region", default="us-east-1")

    args = parser.parse_args()

    engineer = SkillFeatureEngineer(
        database=args.database, region=args.region
    )
    path = engineer.save_features(bucket=args.bucket)
    print(f"Features saved to: {path}")
