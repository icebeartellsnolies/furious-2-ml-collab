"""Stage 1: Prepare data pipeline stage."""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path when running script directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_data, split_data
from src.utils import load_params, set_seed

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def prepare_stage(config_path: str = "params.yaml") -> None:
    """Run data ingestion and split into deterministic train/test CSV datasets."""
    params = load_params(config_path)

    seed = params.get("seed", 42)
    set_seed(seed)

    data_cfg = params.get("data", {})
    raw_path = data_cfg.get("raw_path", "data/raw/dataset.csv")
    processed_dir = Path(data_cfg.get("processed_dir", "data/processed"))
    target_col = data_cfg.get("target_col", "shares")
    test_size = data_cfg.get("test_size", 0.2)
    split_seed = data_cfg.get("random_state", seed)

    logger.info("Stage 1 (Prepare): Ingesting raw dataset from %s...", raw_path)
    df = load_data(raw_path)

    logger.info(
        "Splitting data into train/test sets (test_size=%.2f, random_state=%d)...",
        test_size,
        split_seed,
    )
    X_train, X_test, y_train, y_test = split_data(
        df, target_col=target_col, test_size=test_size, random_state=split_seed
    )

    # Combine features and target back into complete train/test DataFrames
    train_df = X_train.copy()
    train_df[target_col] = y_train

    test_df = X_test.copy()
    test_df[target_col] = y_test

    processed_dir.mkdir(parents=True, exist_ok=True)
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(
        "Successfully exported train.csv (%d samples) and test.csv (%d samples) to %s",
        len(train_df),
        len(test_df),
        processed_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Prepare data pipeline stage")
    parser.add_argument(
        "--config", type=str, default="params.yaml", help="Path to params.yaml"
    )
    args = parser.parse_args()
    prepare_stage(config_path=args.config)
