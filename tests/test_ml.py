"""
Tests for the ML training pipeline and inference module.

All tests use synthetic data so no AWS credentials or S3 access is needed.
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_weekly_data(n_skills: int = 5, n_weeks: int = 30) -> pd.DataFrame:
    """Generate synthetic weekly skill demand data."""
    rng = np.random.default_rng(42)
    skills = [f"skill_{i}" for i in range(n_skills)]

    rows = []
    for skill in skills:
        base = rng.integers(10, 100)
        trend = rng.uniform(-0.5, 1.5)
        for w in range(n_weeks):
            noise = rng.normal(0, 3)
            rows.append({
                "skill": skill,
                "week": pd.Timestamp("2024-01-01") + pd.Timedelta(weeks=w),
                "country": "US",
                "job_count": max(1, int(base + trend * w + noise)),
                "avg_salary": rng.uniform(60_000, 180_000),
            })

    return pd.DataFrame(rows)


def _make_training_parquet(tmp_dir: str) -> str:
    """Create a parquet file with features + target, ready for SkillDemandForecaster."""
    from ml.training.feature_engineering import SkillFeatureEngineer

    raw = _make_weekly_data(n_skills=12, n_weeks=40)
    engineer = SkillFeatureEngineer()

    features_df, target_df = engineer.prepare_training_data(
        target_horizon=4, raw_df=raw
    )
    training = features_df.merge(target_df, on=["skill", "week"])

    path = os.path.join(tmp_dir, "training_data.parquet")
    training.to_parquet(path, index=False)
    return path


# ── Feature Engineering Tests ────────────────────────────────────────────────


class TestFeatureEngineering:
    """Tests for SkillFeatureEngineer."""

    def test_create_time_features(self):
        from ml.training.feature_engineering import SkillFeatureEngineer

        eng = SkillFeatureEngineer()
        raw = _make_weekly_data(n_skills=2, n_weeks=10)
        result = eng.create_time_features(raw)

        for col in ["week_sin", "week_cos", "month_sin", "month_cos", "month", "year"]:
            assert col in result.columns, f"Missing column: {col}"

        assert result["week_sin"].between(-1, 1).all()
        assert result["week_cos"].between(-1, 1).all()

    def test_create_lag_features(self):
        from ml.training.feature_engineering import SkillFeatureEngineer

        eng = SkillFeatureEngineer()
        raw = _make_weekly_data(n_skills=2, n_weeks=20)
        result = eng.create_lag_features(raw)

        for lag in [1, 2, 4, 8, 12]:
            assert f"job_count_lag_{lag}w" in result.columns
            assert f"salary_lag_{lag}w" in result.columns

    def test_create_growth_features(self):
        from ml.training.feature_engineering import SkillFeatureEngineer

        eng = SkillFeatureEngineer()
        raw = _make_weekly_data(n_skills=2, n_weeks=20)
        result = eng.create_growth_features(raw)

        for col in ["wow_growth", "mom_growth", "qoq_growth", "market_share"]:
            assert col in result.columns

    def test_create_rolling_features(self):
        from ml.training.feature_engineering import SkillFeatureEngineer

        eng = SkillFeatureEngineer()
        raw = _make_weekly_data(n_skills=2, n_weeks=20)
        result = eng.create_rolling_features(raw)

        for w in [4, 8, 12]:
            assert f"job_count_ma_{w}w" in result.columns
            assert f"job_count_std_{w}w" in result.columns

    def test_prepare_training_data_from_raw(self):
        from ml.training.feature_engineering import SkillFeatureEngineer

        eng = SkillFeatureEngineer()
        raw = _make_weekly_data(n_skills=4, n_weeks=30)

        features_df, target_df = eng.prepare_training_data(
            target_horizon=4, raw_df=raw
        )

        assert len(features_df) > 0
        assert "target" in target_df.columns
        assert "skill" in features_df.columns
        assert len(features_df) == len(target_df)

        for col in SkillFeatureEngineer.FEATURE_COLS:
            assert col in features_df.columns, f"Missing feature: {col}"

    def test_aggregate_to_weekly(self):
        from ml.training.feature_engineering import SkillFeatureEngineer

        raw_rows = [
            {"skill": "python", "posted_date": "2024-01-15", "country": "US",
             "job_id": "1", "salary_mid": 120_000},
            {"skill": "python", "posted_date": "2024-01-16", "country": "US",
             "job_id": "2", "salary_mid": 130_000},
            {"skill": "react", "posted_date": "2024-01-15", "country": "US",
             "job_id": "3", "salary_mid": 110_000},
        ]
        raw = pd.DataFrame(raw_rows)

        result = SkillFeatureEngineer._aggregate_to_weekly(raw)

        assert "job_count" in result.columns
        assert "avg_salary" in result.columns
        assert len(result) >= 2  # at least python + react


# ── Training Tests ───────────────────────────────────────────────────────────


class TestTraining:
    """Tests for SkillDemandForecaster."""

    def test_full_training_pipeline(self):
        from ml.training.train import SkillDemandForecaster

        with tempfile.TemporaryDirectory() as tmp:
            parquet_path = _make_training_parquet(tmp)

            forecaster = SkillDemandForecaster()
            metrics = forecaster.train(parquet_path)

            assert "demand_model" in metrics
            assert "emergence_model" in metrics
            assert "cluster_model" in metrics
            assert metrics["n_samples"] > 0
            assert metrics["n_skills"] > 0
            assert metrics["n_features"] > 0

    def test_demand_model_metrics(self):
        from ml.training.train import SkillDemandForecaster

        with tempfile.TemporaryDirectory() as tmp:
            parquet_path = _make_training_parquet(tmp)

            forecaster = SkillDemandForecaster()
            metrics = forecaster.train(parquet_path)

            dm = metrics["demand_model"]
            assert "cv_mae_mean" in dm
            assert "cv_r2_mean" in dm
            assert "top_features" in dm
            assert dm["cv_mae_mean"] >= 0
            assert len(dm["top_features"]) > 0

    def test_emergence_detector(self):
        from ml.training.train import SkillDemandForecaster

        with tempfile.TemporaryDirectory() as tmp:
            parquet_path = _make_training_parquet(tmp)

            forecaster = SkillDemandForecaster()
            metrics = forecaster.train(parquet_path)

            em = metrics["emergence_model"]
            assert "n_emerging_detected" in em
            assert "anomaly_threshold" in em
            assert em["n_emerging_detected"] >= 0

    def test_cluster_model(self):
        from ml.training.train import SkillDemandForecaster

        with tempfile.TemporaryDirectory() as tmp:
            parquet_path = _make_training_parquet(tmp)

            forecaster = SkillDemandForecaster()
            metrics = forecaster.train(parquet_path)

            cm = metrics["cluster_model"]
            assert cm["n_clusters"] == 8
            assert "cluster_profiles" in cm
            assert "skill_cluster_mapping" in cm

    def test_save_and_load(self):
        from ml.training.train import SkillDemandForecaster

        with tempfile.TemporaryDirectory() as tmp:
            parquet_path = _make_training_parquet(tmp)
            model_dir = os.path.join(tmp, "models")

            forecaster = SkillDemandForecaster()
            forecaster.train(parquet_path)
            forecaster.save(model_dir)

            expected_files = [
                "demand_model.pkl",
                "emergence_model.pkl",
                "cluster_model.pkl",
                "scaler.pkl",
                "skill_encoder.pkl",
                "feature_cols.json",
                "metrics.json",
            ]
            for fname in expected_files:
                assert (Path(model_dir) / fname).exists(), f"Missing: {fname}"

            loaded = SkillDemandForecaster.load(model_dir)
            assert len(loaded.feature_cols) > 0
            assert loaded.demand_model is not None
            assert loaded.emergence_model is not None
            assert loaded.cluster_model is not None

    def test_metrics_json_valid(self):
        from ml.training.train import SkillDemandForecaster

        with tempfile.TemporaryDirectory() as tmp:
            parquet_path = _make_training_parquet(tmp)
            model_dir = os.path.join(tmp, "models")

            forecaster = SkillDemandForecaster()
            forecaster.train(parquet_path)
            forecaster.save(model_dir)

            with open(os.path.join(model_dir, "metrics.json")) as f:
                data = json.load(f)

            assert "demand_model" in data
            assert "n_samples" in data


# ── Inference Tests ──────────────────────────────────────────────────────────


class TestInference:
    """Tests for SkillDemandPredictor using a locally trained model."""

    @pytest.fixture(scope="class")
    def trained_model_dir(self, tmp_path_factory):
        """Train a model once and share across test methods."""
        from ml.training.train import SkillDemandForecaster

        tmp = tmp_path_factory.mktemp("model")
        parquet_path = str(tmp / "training_data.parquet")

        from ml.training.feature_engineering import SkillFeatureEngineer

        raw = _make_weekly_data(n_skills=12, n_weeks=40)
        eng = SkillFeatureEngineer()
        features_df, target_df = eng.prepare_training_data(target_horizon=4, raw_df=raw)
        training = features_df.merge(target_df, on=["skill", "week"])
        training.to_parquet(parquet_path, index=False)

        model_dir = str(tmp / "models")
        forecaster = SkillDemandForecaster()
        forecaster.train(parquet_path)
        forecaster.save(model_dir)

        return model_dir

    def test_predictor_loads(self, trained_model_dir):
        from ml.inference.predictor import SkillDemandPredictor

        predictor = SkillDemandPredictor(model_path=trained_model_dir)
        assert predictor.feature_cols
        assert predictor.demand_model is not None

    def test_predictor_has_metrics(self, trained_model_dir):
        from ml.inference.predictor import SkillDemandPredictor

        predictor = SkillDemandPredictor(model_path=trained_model_dir)
        assert isinstance(predictor.metrics, dict)
        assert "demand_model" in predictor.metrics

    def test_predictor_cluster_mapping(self, trained_model_dir):
        from ml.inference.predictor import SkillDemandPredictor

        predictor = SkillDemandPredictor(model_path=trained_model_dir)
        mapping = predictor.metrics.get("cluster_model", {}).get("skill_cluster_mapping", {})
        assert len(mapping) > 0


# ── Adzuna Scraper Tests ────────────────────────────────────────────────────


class TestAdzunaScraper:
    """Unit tests for AdzunaScraper static methods (no AWS or API needed)."""

    def test_extract_skills_basic(self):
        from ingestion.scrapers.adzuna_scraper import AdzunaScraper

        skills = AdzunaScraper.extract_skills(
            "We need Python and AWS experience with Kubernetes"
        )
        assert "python" in skills
        assert "aws" in skills
        assert "kubernetes" in skills

    def test_extract_skills_empty(self):
        from ingestion.scrapers.adzuna_scraper import AdzunaScraper

        assert AdzunaScraper.extract_skills("") == []
        assert AdzunaScraper.extract_skills(None) == []

    def test_extract_skills_multiword(self):
        from ingestion.scrapers.adzuna_scraper import AdzunaScraper

        skills = AdzunaScraper.extract_skills(
            "deep learning and machine learning with computer vision focus"
        )
        assert "deep_learning" in skills
        assert "machine_learning" in skills
        assert "computer_vision" in skills

    def test_generate_job_id(self):
        from ingestion.scrapers.adzuna_scraper import AdzunaScraper

        job_with_id = {"id": "12345"}
        assert AdzunaScraper.generate_job_id(job_with_id) == "adzuna_12345"

        job_without_id = {
            "title": "Software Engineer",
            "company": {"display_name": "Acme"},
            "location": {"display_name": "NYC"},
        }
        result = AdzunaScraper.generate_job_id(job_without_id)
        assert result.startswith("adzuna_")
        assert len(result) > 7
