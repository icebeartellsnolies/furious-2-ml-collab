"""Feature engineering and transformation utilities for OnlineNewsPopularity dataset."""

import numpy as np
import pandas as pd


def create_engagement_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate domain-specific engagement and interaction features from raw news article attributes.

    Engineered features:
    - content_to_link_ratio: Word count relative to number of external hyperlinks.
    - visual_media_count: Total combined count of images and videos.
    - title_sentiment_interaction: Interaction term between title sentiment polarity and subjectivity.
    - headline_word_ratio: Ratio of title tokens relative to total content tokens.
    """
    df = df.copy()

    # Content to link ratio (add epsilon to prevent division by zero)
    if "n_tokens_content" in df.columns and "num_hrefs" in df.columns:
        df["content_to_link_ratio"] = df["n_tokens_content"] / (df["num_hrefs"] + 1.0)

    # Combined multimedia assets
    if "num_imgs" in df.columns and "num_videos" in df.columns:
        df["visual_media_count"] = df["num_imgs"] + df["num_videos"]

    # Sentiment interaction: strong subjectivity amplifies sentiment polarity
    if "title_sentiment_polarity" in df.columns and "title_subjectivity" in df.columns:
        df["title_sentiment_interaction"] = (
            df["title_sentiment_polarity"] * df["title_subjectivity"]
        )

    # Headline vs article density
    if "n_tokens_title" in df.columns and "n_tokens_content" in df.columns:
        df["headline_word_ratio"] = df["n_tokens_title"] / (
            df["n_tokens_content"] + 1.0
        )

    return df


def calculate_log_target(df: pd.DataFrame, target_col: str = "shares") -> pd.Series:
    """Compute log1p transformation of target variable to normalize highly skewed social share distribution."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")
    return np.log1p(df[target_col])


def cap_outliers(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.DataFrame:
    """Cap numerical feature outliers at specified percentiles."""
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col in df.columns:
            lower = df[col].quantile(lower_quantile)
            upper = df[col].quantile(upper_quantile)
            df[col] = df[col].clip(lower=lower, upper=upper)
    return df
