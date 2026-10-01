# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 01 - Exploratory Data Analysis (OnlineNewsPopularity)
#
# **Project:** MLOps Assignment 01 - Furious-2 ML Collaboration
# **Dataset:** Online News Popularity (UCI Machine Learning Repository)
# **Goal:** Analyze article features, target distribution (`shares`), and demonstrate feature engineering.

# %%
import sys
from pathlib import Path

# Ensure project root is in Python path
PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_data
from src.features import calculate_log_target, create_engagement_features

# %% [markdown]
# ## 1. Data Ingestion & Initial Inspection

# %%
df = load_data("../data/raw/OnlineNewsPopularity.csv")
print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
df.head()

# %% [markdown]
# ## 2. Target Variable Analysis (`shares`)

# %%
shares_summary = df["shares"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
print("Target 'shares' Summary Statistics:")
print(shares_summary)

# Calculate log1p transformed target using src.features
log_shares = calculate_log_target(df, target_col="shares")
print(f"\nOriginal Target Skewness: {df['shares'].skew():.4f}")
print(f"Log-Transformed Target Skewness: {log_shares.skew():.4f}")

# %% [markdown]
# ## 3. Data Channel Distribution Analysis

# %%
channel_cols = [c for c in df.columns if c.startswith("data_channel_is_")]
channel_counts = df[channel_cols].sum().sort_values(ascending=False)
print("Article Count per Data Channel:")
print(channel_counts)

# %% [markdown]
# ## 4. Feature Engineering Demonstration (`src.features`)

# %%
# Apply reusable feature engineering pipeline from src/features.py
df_featured = create_engagement_features(df)
engineered_cols = [
    "content_to_link_ratio",
    "visual_media_count",
    "title_sentiment_interaction",
    "headline_word_ratio",
]
print("Engineered Features Sample:")
print(df_featured[engineered_cols].head())

# %% [markdown]
# ## 5. Key EDA Findings & Insights for Modeling
#
# 1. **Target Skewness:** The target `shares` exhibits high positive skewness (few viral articles have >100,000 shares while median is ~1,400). Log transformation (`log1p`) effectively normalizes the target distribution.
# 2. **Channel Impact:** Articles belong to distinct categories (Tech, World, Entertainment, Business, Social Media, Lifestyle). Category flags provide strong predictive signals.
# 3. **Engineered Features:** Features such as `visual_media_count` (images + videos) and `content_to_link_ratio` capture content richness effectively.
