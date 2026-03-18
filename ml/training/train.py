"""
SageMaker Training Script: Skill Demand Forecasting

Models trained:
1. XGBoost       - Primary demand forecaster
2. Isolation Forest - Emerging skill detector
3. KMeans        - Skill clustering

Usage (local — from parquet):
    python -m ml.training.train --train data/training_data.parquet --model-dir ml/models

Usage (local — from S3):
    python -m ml.training.train --bucket my-bucket --model-dir ml/models

Usage (SageMaker):
    Automatically receives --train, --model-dir, --output-data-dir via SM env vars.
"""

import argparse
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import awswrangler as wr
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SkillDemandForecaster:
    """Multi-model forecasting system for skill demand prediction."""

    EMERGENCE_FEATURES = [
        "mom_growth",
        "qoq_growth",
        "growth_acceleration",
        "market_share_change",
        "trend_strength_8w",
        "emergence_score",
    ]

    def __init__(self):
        self.demand_model = None
        self.emergence_model = None
        self.cluster_model = None
        self.scaler = StandardScaler()
        self.skill_encoder = LabelEncoder()
        self.feature_cols: List[str] = []
        self.metrics: Dict[str, Any] = {}

    # ── data prep ───────────────────────────────────────────────────────────

    def prepare_data(self, df: pd.DataFrame):
        df["skill_encoded"] = self.skill_encoder.fit_transform(df["skill"])

        exclude = {"skill", "week", "country", "target", "skill_encoded"}
        self.feature_cols = [c for c in df.columns if c not in exclude]

        X = self.scaler.fit_transform(df[self.feature_cols].values)
        y = df["target"].values

        return X, y, df

    # ── model 1: XGBoost demand forecaster ──────────────────────────────────

    def train_demand_model(self, X: np.ndarray, y: np.ndarray) -> Dict:
        import xgboost as xgb

        logger.info("Training XGBoost demand model …")

        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores: Dict[str, list] = {"mae": [], "rmse": [], "r2": []}

        xgb_params = dict(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            model = xgb.XGBRegressor(**xgb_params)
            model.fit(
                X[train_idx], y[train_idx],
                eval_set=[(X[val_idx], y[val_idx])],
                verbose=False,
            )
            preds = model.predict(X[val_idx])

            cv_scores["mae"].append(mean_absolute_error(y[val_idx], preds))
            cv_scores["rmse"].append(
                np.sqrt(mean_squared_error(y[val_idx], preds))
            )
            cv_scores["r2"].append(r2_score(y[val_idx], preds))
            logger.info(
                "  Fold %d  MAE=%.2f  R²=%.3f",
                fold, cv_scores["mae"][-1], cv_scores["r2"][-1],
            )

        self.demand_model = xgb.XGBRegressor(**xgb_params)
        self.demand_model.fit(X, y, verbose=False)

        importance = dict(
            zip(self.feature_cols, self.demand_model.feature_importances_)
        )
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

        metrics = {
            "cv_mae_mean": float(np.mean(cv_scores["mae"])),
            "cv_mae_std": float(np.std(cv_scores["mae"])),
            "cv_rmse_mean": float(np.mean(cv_scores["rmse"])),
            "cv_r2_mean": float(np.mean(cv_scores["r2"])),
            "top_features": [(f, float(v)) for f, v in top_features],
        }
        logger.info(
            "  CV MAE: %.2f ± %.2f   Top features: %s",
            metrics["cv_mae_mean"],
            metrics["cv_mae_std"],
            [f for f, _ in top_features[:5]],
        )
        return metrics

    # ── model 2: Isolation Forest emergence detector ────────────────────────

    def train_emergence_detector(
        self, X: np.ndarray, df: pd.DataFrame
    ) -> Dict:
        logger.info("Training Isolation Forest emergence detector …")

        eidx = [
            self.feature_cols.index(f)
            for f in self.EMERGENCE_FEATURES
            if f in self.feature_cols
        ]
        X_em = X[:, eidx]

        self.emergence_model = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42,
            n_jobs=-1,
        )
        self.emergence_model.fit(X_em)

        scores = self.emergence_model.decision_function(X_em)
        preds = self.emergence_model.predict(X_em)

        tmp = df.copy()
        tmp["anomaly_score"] = -scores
        tmp["is_emerging"] = (preds == -1) & (tmp.get("mom_growth", 0) > 0.1)

        emerging_skills = tmp.loc[tmp["is_emerging"], "skill"].unique()

        metrics = {
            "n_emerging_detected": int(len(emerging_skills)),
            "emerging_skills_sample": sorted(emerging_skills)[:10],
            "anomaly_threshold": float(np.percentile(-scores, 90)),
        }
        logger.info("  Detected %d potentially emerging skills", len(emerging_skills))
        return metrics

    # ── model 3: KMeans skill clusters ──────────────────────────────────────

    def train_skill_clusters(
        self, X: np.ndarray, df: pd.DataFrame, n_clusters: int = 8
    ) -> Dict:
        logger.info("Training KMeans skill clusters (k=%d) …", n_clusters)

        skill_features = df.groupby("skill")[self.feature_cols].mean()
        X_skills = self.scaler.fit_transform(skill_features.values)

        self.cluster_model = KMeans(
            n_clusters=n_clusters, random_state=42, n_init=10
        )
        labels = self.cluster_model.fit_predict(X_skills)

        skill_clusters = dict(zip(skill_features.index, (int(l) for l in labels)))

        profiles: Dict[int, Dict] = {}
        for cid in range(n_clusters):
            members = [s for s, c in skill_clusters.items() if c == cid]
            profiles[cid] = {"skills": members, "count": len(members)}

        metrics = {
            "n_clusters": n_clusters,
            "cluster_profiles": profiles,
            "skill_cluster_mapping": skill_clusters,
        }
        for cid, p in profiles.items():
            logger.info("  Cluster %d (%d skills): %s …", cid, p["count"], p["skills"][:5])
        return metrics

    # ── main entry ──────────────────────────────────────────────────────────

    def train(self, train_path: str) -> Dict:
        logger.info("Loading training data from %s", train_path)

        if train_path.endswith(".parquet"):
            df = pd.read_parquet(train_path)
        else:
            df = pd.read_csv(train_path)

        logger.info(
            "Loaded %d samples for %d skills", len(df), df["skill"].nunique()
        )

        X, y, df = self.prepare_data(df)

        demand_m = self.train_demand_model(X, y)
        emerge_m = self.train_emergence_detector(X, df)
        cluster_m = self.train_skill_clusters(X, df)

        self.metrics = {
            "demand_model": demand_m,
            "emergence_model": emerge_m,
            "cluster_model": cluster_m,
            "n_samples": len(df),
            "n_skills": int(df["skill"].nunique()),
            "n_features": len(self.feature_cols),
        }
        return self.metrics

    # ── persistence ─────────────────────────────────────────────────────────

    def save(self, model_dir: str) -> None:
        os.makedirs(model_dir, exist_ok=True)

        artifacts = {
            "demand_model.pkl": self.demand_model,
            "emergence_model.pkl": self.emergence_model,
            "cluster_model.pkl": self.cluster_model,
            "scaler.pkl": self.scaler,
            "skill_encoder.pkl": self.skill_encoder,
        }
        for fname, obj in artifacts.items():
            with open(os.path.join(model_dir, fname), "wb") as f:
                pickle.dump(obj, f)

        with open(os.path.join(model_dir, "feature_cols.json"), "w") as f:
            json.dump(self.feature_cols, f)

        with open(os.path.join(model_dir, "metrics.json"), "w") as f:
            json.dump(self.metrics, f, indent=2, default=str)

        logger.info("Models saved to %s", model_dir)

    @classmethod
    def load(cls, model_dir: str) -> "SkillDemandForecaster":
        forecaster = cls()

        blobs = [
            ("demand_model.pkl", "demand_model"),
            ("emergence_model.pkl", "emergence_model"),
            ("cluster_model.pkl", "cluster_model"),
            ("scaler.pkl", "scaler"),
            ("skill_encoder.pkl", "skill_encoder"),
        ]
        for fname, attr in blobs:
            with open(os.path.join(model_dir, fname), "rb") as f:
                setattr(forecaster, attr, pickle.load(f))

        with open(os.path.join(model_dir, "feature_cols.json")) as f:
            forecaster.feature_cols = json.load(f)

        return forecaster

    # ── S3 integration ──────────────────────────────────────────────────────

    def upload_to_s3(self, model_dir: str, bucket: str) -> str:
        """Upload all model artifacts from *model_dir* to S3."""
        s3_prefix = f"s3://{bucket}/models/skill_forecaster/"

        for fpath in Path(model_dir).glob("*"):
            if fpath.is_file():
                wr.s3.upload(
                    local_file=str(fpath),
                    path=f"{s3_prefix}{fpath.name}",
                )

        logger.info("Models uploaded to %s", s3_prefix)
        return s3_prefix

    @classmethod
    def train_from_s3(
        cls,
        bucket: str,
        data_source: str = "adzuna",
        database: str = "job_market_db",
        region: str = "us-east-1",
        model_dir: str = "ml/models",
        upload_s3: bool = False,
    ) -> "SkillDemandForecaster":
        """
        End-to-end pipeline: load from S3 → feature engineering → train → save.

        Args:
            bucket: S3 bucket name.
            data_source: 'adzuna' (only supported source).
            database: Glue catalog database name.
            region: AWS region.
            model_dir: local directory for saving model artifacts.
            upload_s3: whether to also push artifacts to S3.

        Returns:
            Trained SkillDemandForecaster instance.
        """
        from ml.training.feature_engineering import SkillFeatureEngineer

        logger.info("=" * 60)
        logger.info("SKILL DEMAND FORECASTING — END-TO-END PIPELINE")
        logger.info("=" * 60)
        logger.info("Data source : %s", data_source)
        logger.info("Bucket      : %s", bucket)

        engineer = SkillFeatureEngineer(database=database, region=region)
        features_df, target_df = engineer.prepare_training_data_from_s3(
            bucket=bucket, data_source=data_source
        )

        training_data = features_df.merge(target_df, on=["skill", "week"])

        temp_path = os.path.join(model_dir, "_training_data.parquet")
        os.makedirs(model_dir, exist_ok=True)
        training_data.to_parquet(temp_path, index=False)
        logger.info("Training data cached at %s (%d rows)", temp_path, len(training_data))

        forecaster = cls()
        metrics = forecaster.train(temp_path)
        forecaster.save(model_dir)

        if upload_s3:
            forecaster.upload_to_s3(model_dir, bucket)

        os.remove(temp_path)

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)

        return forecaster


# ── SageMaker hosting hooks ────────────────────────────────────────────────


def model_fn(model_dir: str):
    """SageMaker model loading function."""
    return SkillDemandForecaster.load(model_dir)


def predict_fn(input_data: pd.DataFrame, model: SkillDemandForecaster):
    """SageMaker prediction function."""

    X = model.scaler.transform(input_data[model.feature_cols].values)

    demand_preds = model.demand_model.predict(X)

    eidx = [
        model.feature_cols.index(f)
        for f in model.EMERGENCE_FEATURES
        if f in model.feature_cols
    ]
    emergence_scores = -model.emergence_model.decision_function(X[:, eidx])

    return {
        "demand_forecast": demand_preds.tolist(),
        "emergence_score": emergence_scores.tolist(),
    }


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train skill demand forecasting models"
    )

    parser.add_argument(
        "--train",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN"),
        help="Path / S3 URI to training parquet (skip if using --bucket)",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        help="S3 bucket — enables end-to-end pipeline (load → featurize → train)",
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default="adzuna",
        choices=["adzuna"],
        help="Data source to load from S3",
    )
    parser.add_argument("--database", default="job_market_db")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument(
        "--upload-s3",
        action="store_true",
        help="Upload model artifacts to S3 after training",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "ml/models"),
    )
    parser.add_argument(
        "--output-data-dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "ml/output"),
    )

    args = parser.parse_args()

    if args.bucket:
        forecaster = SkillDemandForecaster.train_from_s3(
            bucket=args.bucket,
            data_source=args.data_source,
            database=args.database,
            region=args.region,
            model_dir=args.model_dir,
            upload_s3=args.upload_s3,
        )
        metrics = forecaster.metrics
    elif args.train:
        forecaster = SkillDemandForecaster()
        metrics = forecaster.train(args.train)
        forecaster.save(args.model_dir)
        if args.upload_s3 and args.bucket:
            forecaster.upload_to_s3(args.model_dir, args.bucket)
    else:
        parser.error("Provide either --train <path> or --bucket <s3-bucket>")

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Samples            : {metrics['n_samples']}")
    print(f"  Skills             : {metrics['n_skills']}")
    print(f"  Features           : {metrics['n_features']}")
    print(f"  Demand Model R²    : {metrics['demand_model']['cv_r2_mean']:.3f}")
    print(f"  Emerging Detected  : {metrics['emergence_model']['n_emerging_detected']}")
    print(f"  Skill Clusters     : {metrics['cluster_model']['n_clusters']}")
    print(f"  Models saved to    : {args.model_dir}")
    if args.upload_s3 and args.bucket:
        print(f"  Models on S3       : s3://{args.bucket}/models/skill_forecaster/")
    print("=" * 60)
