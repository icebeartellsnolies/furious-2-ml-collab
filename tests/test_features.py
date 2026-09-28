import numpy as np
import pandas as pd

from src.features import (
    calculate_log_target,
    cap_outliers,
    create_engagement_features,
)


def test_create_engagement_features():
    df = pd.DataFrame(
        {
            "n_tokens_title": [10, 15],
            "n_tokens_content": [500, 1000],
            "num_hrefs": [4, 9],
            "num_imgs": [2, 5],
            "num_videos": [1, 0],
            "title_sentiment_polarity": [0.5, -0.2],
            "title_subjectivity": [0.8, 0.4],
            "shares": [1200, 3500],
        }
    )

    df_featured = create_engagement_features(df)

    assert "content_to_link_ratio" in df_featured.columns
    assert "visual_media_count" in df_featured.columns
    assert "title_sentiment_interaction" in df_featured.columns
    assert "headline_word_ratio" in df_featured.columns

    # Verify calculation: visual_media_count = num_imgs + num_videos
    assert df_featured.loc[0, "visual_media_count"] == 3
    assert df_featured.loc[1, "visual_media_count"] == 5

    # Verify interaction: 0.5 * 0.8 = 0.4
    assert np.isclose(df_featured.loc[0, "title_sentiment_interaction"], 0.4)


def test_calculate_log_target():
    df = pd.DataFrame({"shares": [0, 99, 999]})
    log_shares = calculate_log_target(df, target_col="shares")

    assert len(log_shares) == 3
    assert np.isclose(log_shares.iloc[0], np.log1p(0))
    assert np.isclose(log_shares.iloc[1], np.log1p(99))


def test_cap_outliers():
    df = pd.DataFrame({"val": [1, 50, 50, 50, 1000]})
    df_capped = cap_outliers(
        df, columns=["val"], lower_quantile=0.1, upper_quantile=0.9
    )

    assert df_capped["val"].min() > 1
    assert df_capped["val"].max() < 1000
