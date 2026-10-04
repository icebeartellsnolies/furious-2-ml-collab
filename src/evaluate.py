"""Stage 3: Evaluate pipeline stage."""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path when running script directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features import create_engagement_features
from src.utils import get_git_commit_sha, load_params, save_metrics

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def evaluate_stage(config_path: str = "params.yaml") -> dict:
    """Load test split and model artifact, evaluate model metrics, and export metrics.json."""
    params = load_params(config_path)

    seed = params.get("seed", 42)
    data_cfg = params.get("data", {})
    processed_dir = Path(data_cfg.get("processed_dir", "data/processed"))
    target_col = data_cfg.get("target_col", "shares")

    eval_cfg = params.get("evaluate", {})
    metrics_path = eval_cfg.get("metrics_file", "metrics.json")
    train_cfg = params.get("train", {})

    test_path = processed_dir / "test.csv"
    model_path = Path("models/model.joblib")

    logger.info("Stage 3 (Evaluate): Loading test set from %s...", test_path)
    if not test_path.exists():
        raise FileNotFoundError(f"Processed test file not found at {test_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}")

    test_df = pd.read_csv(test_path)
    if target_col not in test_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in test dataframe.")

    y_test = test_df[target_col]
    X_test_raw = test_df.drop(columns=[target_col])

    # Load serialized model artifact (contains model, pre-fitted scaler, and expected feature list)
    artifact = joblib.load(model_path)
    model = artifact["model"]
    scaler = artifact["scaler"]
    expected_features = artifact["features"]

    # Apply domain feature engineering
    X_test_featured = create_engagement_features(X_test_raw)

    # Ensure feature alignment with training schema
    X_test_aligned = X_test_featured[expected_features]

    # Transform test features using the pre-fitted scaler from training split
    X_test_scaled = scaler.transform(X_test_aligned)

    # Compute evaluation metrics
    y_pred = model.predict(X_test_scaled)
    log_target = artifact.get("log_target", False)
    if log_target:
        y_pred = np.expm1(y_pred)  # metrics stay on the raw shares scale
    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))

    commit_sha = get_git_commit_sha()

    metrics = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "commit_sha": commit_sha,
        "seed": seed,
        "n_test_samples": len(test_df),
        "hyperparameters": {
            "model_type": train_cfg.get("model_type", "random_forest"),
            "n_estimators": train_cfg.get("n_estimators", 100),
            "max_depth": train_cfg.get("max_depth", 6),
            "log_target": log_target,
        },
    }

    save_metrics(metrics, metrics_path)
    logger.info("Saved metrics to %s", metrics_path)
    logger.info(
        "Evaluation Results -> MAE: %.4f | RMSE: %.4f | R²: %.4f", mae, rmse, r2
    )

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: Evaluate pipeline stage")
    parser.add_argument(
        "--config", type=str, default="params.yaml", help="Path to params.yaml"
    )
    args = parser.parse_args()
    evaluate_stage(config_path=args.config)
