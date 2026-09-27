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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_data, split_data
from src.utils import get_git_commit_sha, load_params, save_metrics, set_seed

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def train_pipeline(config_path: str = "params.yaml") -> dict:
    """Run end-to-end baseline training pipeline."""
    params = load_params(config_path)

    seed = params.get("seed", 42)
    set_seed(seed)

    data_cfg = params.get("data", {})
    raw_path = data_cfg.get("raw_path", "data/raw/OnlineNewsPopularity.csv")
    target_col = data_cfg.get("target_col", "shares")
    test_size = data_cfg.get("test_size", 0.2)
    split_seed = data_cfg.get("random_state", seed)

    train_cfg = params.get("train", {})
    n_estimators = train_cfg.get("n_estimators", 100)
    max_depth = train_cfg.get("max_depth", 6)
    model_seed = train_cfg.get("random_state", seed)

    eval_cfg = params.get("evaluate", {})
    metrics_path = eval_cfg.get("metrics_file", "metrics.json")

    logger.info("Starting baseline training pipeline (OnlineNewsPopularity)...")

    # 1. Load Data
    df = load_data(raw_path)

    # 2. Split Data
    X_train, X_test, y_train, y_test = split_data(
        df, target_col=target_col, test_size=test_size, random_state=split_seed
    )

    # 3. Preprocess Features (Fit on train split only to prevent leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train Model
    logger.info(
        "Training RandomForestRegressor (n_estimators=%d, max_depth=%d, seed=%d)",
        n_estimators,
        max_depth,
        model_seed,
    )
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=model_seed,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    # 5. Evaluate
    y_pred = model.predict(X_test_scaled)
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
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "hyperparameters": {
            "model_type": train_cfg.get("model_type", "random_forest"),
            "n_estimators": n_estimators,
            "max_depth": max_depth,
        },
    }

    # 6. Save Artifacts
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_artifact = {
        "model": model,
        "scaler": scaler,
        "features": list(X_train.columns),
    }
    model_path = models_dir / "baseline_model.joblib"
    joblib.dump(model_artifact, model_path)
    logger.info("Saved model artifact to %s", model_path)

    save_metrics(metrics, metrics_path)
    logger.info("Saved metrics to %s", metrics_path)

    logger.info("Pipeline completed successfully!")
    logger.info(
        "Evaluation Results -> MAE: %.4f | RMSE: %.4f | R²: %.4f", mae, rmse, r2
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train OnlineNewsPopularity baseline model"
    )
    parser.add_argument(
        "--config", type=str, default="params.yaml", help="Path to params.yaml"
    )
    args = parser.parse_args()

    train_pipeline(config_path=args.config)
