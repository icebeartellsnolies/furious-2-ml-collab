"""Stage 2: Train pipeline stage."""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path when running script directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from src.features import create_engagement_features
from src.utils import load_params, set_seed

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def train_stage(config_path: str = "params.yaml") -> None:
    """Load train split, apply domain feature engineering, fit preprocessor on train only, train model, and save artifact."""
    params = load_params(config_path)

    seed = params.get("seed", 42)
    set_seed(seed)

    data_cfg = params.get("data", {})
    processed_dir = Path(data_cfg.get("processed_dir", "data/processed"))
    target_col = data_cfg.get("target_col", "shares")

    train_cfg = params.get("train", {})
    model_type = train_cfg.get("model_type", "random_forest")
    n_estimators = train_cfg.get("n_estimators", 100)
    max_depth = train_cfg.get("max_depth", 6)
    model_seed = train_cfg.get("random_state", seed)

    train_path = processed_dir / "train.csv"
    logger.info("Stage 2 (Train): Loading train split from %s...", train_path)
    if not train_path.exists():
        raise FileNotFoundError(f"Processed training file not found at {train_path}")

    train_df = pd.read_csv(train_path)

    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in train dataset.")

    y_train = train_df[target_col]
    X_train_raw = train_df.drop(columns=[target_col])

    # Apply domain feature engineering
    logger.info("Applying domain feature engineering (create_engagement_features)...")
    X_train_featured = create_engagement_features(X_train_raw)

    # Fit preprocessor strictly on training split to avoid data leakage
    logger.info("Fitting StandardScaler on training features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_featured)
    feature_names = list(X_train_featured.columns)

    # Fit model with fixed seeds
    logger.info(
        "Training %s (n_estimators=%d, max_depth=%d, seed=%d)...",
        model_type,
        n_estimators,
        max_depth,
        model_seed,
    )
    if model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=model_seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported model_type: '{model_type}'")

    model.fit(X_train_scaled, y_train)

    # Serialize model artifact
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_artifact = {
        "model": model,
        "scaler": scaler,
        "features": feature_names,
    }
    model_path = models_dir / "model.joblib"
    joblib.dump(model_artifact, model_path)
    logger.info("Saved trained model artifact to %s", model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Train pipeline stage")
    parser.add_argument(
        "--config", type=str, default="params.yaml", help="Path to params.yaml"
    )
    args = parser.parse_args()
    train_stage(config_path=args.config)
