import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Non-predictive metadata columns in OnlineNewsPopularity
NON_PREDICTIVE_COLS = ["url", "timedelta"]


def generate_synthetic_data(
    num_samples: int = 200, random_state: int = 42
) -> pd.DataFrame:
    """Generate a synthetic DataFrame matching the OnlineNewsPopularity schema for smoke testing/CI."""
    rng = np.random.RandomState(random_state)
    logger.warning(
        "Generating %d synthetic samples for pipeline verification.", num_samples
    )

    features = {
        "n_tokens_title": rng.randint(5, 20, num_samples),
        "n_tokens_content": rng.randint(50, 1000, num_samples),
        "n_unique_tokens": rng.uniform(0.1, 0.9, num_samples),
        "num_hrefs": rng.randint(0, 50, num_samples),
        "num_imgs": rng.randint(0, 10, num_samples),
        "num_videos": rng.randint(0, 5, num_samples),
        "average_token_length": rng.uniform(4.0, 6.0, num_samples),
        "num_keywords": rng.randint(1, 10, num_samples),
        "data_channel_is_lifestyle": rng.choice([0, 1], num_samples),
        "data_channel_is_entertainment": rng.choice([0, 1], num_samples),
        "data_channel_is_bus": rng.choice([0, 1], num_samples),
        "data_channel_is_socmed": rng.choice([0, 1], num_samples),
        "data_channel_is_tech": rng.choice([0, 1], num_samples),
        "data_channel_is_world": rng.choice([0, 1], num_samples),
        "is_weekend": rng.choice([0, 1], num_samples),
        "global_subjectivity": rng.uniform(0.0, 1.0, num_samples),
        "global_sentiment_polarity": rng.uniform(-0.5, 0.5, num_samples),
        "shares": rng.randint(100, 10000, num_samples),
    }
    return pd.DataFrame(features)


def load_data(data_path: str = "data/raw/OnlineNewsPopularity.csv") -> pd.DataFrame:
    """
    Load OnlineNewsPopularity dataset.
    Handles relative paths, column name whitespace stripping, and non-predictive column removal.
    """
    path = Path(data_path)
    if not path.exists():
        alternatives = [
            Path("data/raw/dataset.csv"),
            Path("data/raw/OnlineNewsPopularity.csv"),
        ]
        found = False
        for alt in alternatives:
            if alt.exists():
                path = alt
                found = True
                break
        if not found:
            logger.warning(
                "Raw dataset not found at '%s'. Falling back to synthetic verification dataset.",
                data_path,
            )
            return generate_synthetic_data()

    logger.info("Loading dataset from %s", path)
    df = pd.read_csv(path)

    # Clean whitespace in column names (common in UCI OnlineNewsPopularity dataset)
    df.columns = df.columns.str.strip()

    # Drop non-predictive columns if present
    cols_to_drop = [c for c in NON_PREDICTIVE_COLS if c in df.columns]
    if cols_to_drop:
        logger.info("Dropping non-predictive columns: %s", cols_to_drop)
        df = df.drop(columns=cols_to_drop)

    return df


def split_data(
    df: pd.DataFrame,
    target_col: str = "shares",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split feature matrix and target into train and test splits with fixed seed.
    """
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in dataframe columns: {list(df.columns)}"
        )

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(
        "Split data into train (%d samples) and test (%d samples)",
        len(X_train),
        len(X_test),
    )
    return X_train, X_test, y_train, y_test
