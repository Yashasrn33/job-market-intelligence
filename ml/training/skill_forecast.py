"""
Convenience wrapper: runs only the XGBoost demand forecaster
from the full training pipeline.

For the complete multi-model training (XGBoost + IsolationForest + KMeans)
use ``python -m ml.training.train``.
"""

import logging
from typing import Any, Dict, Tuple

from ml.training.train import SkillDemandForecaster

logger = logging.getLogger(__name__)


def train_forecast_model(
    train_path: str,
    model_dir: str = "ml/models",
) -> Tuple[SkillDemandForecaster, Dict[str, Any]]:
    """
    Train the full SkillDemandForecaster and return it with metrics.

    Args:
        train_path: local or S3 path to the training parquet.
        model_dir: where to persist model artifacts.

    Returns:
        (forecaster, metrics_dict)
    """

    forecaster = SkillDemandForecaster()
    metrics = forecaster.train(train_path)
    forecaster.save(model_dir)

    return forecaster, metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train demand forecast model")
    parser.add_argument("--train", required=True, help="Path to training parquet")
    parser.add_argument("--model-dir", default="ml/models")
    args = parser.parse_args()

    _, m = train_forecast_model(args.train, args.model_dir)
    print(f"Done.  CV R²={m['demand_model']['cv_r2_mean']:.3f}")
