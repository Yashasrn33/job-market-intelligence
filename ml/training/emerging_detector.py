"""
Emerging Skill Detector

Loads the trained Isolation Forest from ml/models and returns skills
whose anomaly score exceeds a threshold (i.e. unusually rapid growth).
"""

import json
import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ml.training.train import SkillDemandForecaster

logger = logging.getLogger(__name__)


def detect_emerging_skills(
    data_path: str,
    model_dir: str = "ml/models",
    top_n: int = 20,
) -> List[Dict[str, Any]]:
    """
    Score skills using the trained IsolationForest and return
    the top-N most anomalous (positively-growing) ones.

    Returns:
        List of dicts with ``skill``, ``anomaly_score``, ``mom_growth``.
    """

    model = SkillDemandForecaster.load(model_dir)

    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    X = model.scaler.transform(df[model.feature_cols].values)

    eidx = [
        model.feature_cols.index(f)
        for f in model.EMERGENCE_FEATURES
        if f in model.feature_cols
    ]
    scores = -model.emergence_model.decision_function(X[:, eidx])

    df["anomaly_score"] = scores
    mom_col = "mom_growth" if "mom_growth" in df.columns else None

    if mom_col:
        df = df[df[mom_col] > 0]

    latest = df.sort_values("week").groupby("skill").tail(1)
    latest = latest.nlargest(top_n, "anomaly_score")

    results = []
    for _, row in latest.iterrows():
        results.append({
            "skill": row["skill"],
            "anomaly_score": float(row["anomaly_score"]),
            "mom_growth": float(row.get(mom_col, 0)) if mom_col else None,
        })

    logger.info("Top %d emerging skills: %s", len(results), [r["skill"] for r in results])
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect emerging skills")
    parser.add_argument("--data", required=True, help="Path to feature parquet")
    parser.add_argument("--model-dir", default="ml/models")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    results = detect_emerging_skills(args.data, args.model_dir, args.top_n)
    print(json.dumps(results, indent=2))
