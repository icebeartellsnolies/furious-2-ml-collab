"""Data checks: schema, null counts and value ranges. Exit code 1 on any failure."""

import argparse
import sys

import pandas as pd

TARGET = "shares"
REQUIRED_COLS = [
    "n_tokens_title",
    "n_tokens_content",
    "num_hrefs",
    "num_imgs",
    "num_videos",
    "global_subjectivity",
    "global_sentiment_polarity",
    "is_weekend",
    TARGET,
]
BINARY_COLS = ["is_weekend"]  # plus every data_channel_is_* / weekday_is_* column
UNIT_RANGE_COLS = [
    "global_subjectivity",
    "LDA_00",
    "LDA_01",
    "LDA_02",
    "LDA_03",
    "LDA_04",
]
NON_NEGATIVE_COLS = [
    "n_tokens_title",
    "n_tokens_content",
    "num_hrefs",
    "num_imgs",
    "num_videos",
]


def check_data(df: pd.DataFrame) -> list[str]:
    """Return a list of human-readable problems (empty list = data is fine)."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    errors = []

    # Schema
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        errors.append(f"missing columns: {missing}")
    non_numeric = [
        c for c in df.columns if c != "url" and not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        errors.append(f"non-numeric columns: {non_numeric}")

    # Nulls
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0]
    if not nulls.empty:
        errors.append(f"null counts: {nulls.to_dict()}")

    # Ranges (only on columns that exist, so a schema error isn't reported twice)
    binary = BINARY_COLS + [
        c for c in df.columns if c.startswith(("data_channel_is_", "weekday_is_"))
    ]
    for c in binary:
        if c in df.columns and not df[c].isin([0, 1]).all():
            errors.append(f"{c} must be 0/1")
    for c in UNIT_RANGE_COLS:
        if c in df.columns and not df[c].between(0, 1).all():
            errors.append(f"{c} must be within [0, 1]")
    for c in NON_NEGATIVE_COLS:
        if c in df.columns and not (df[c] >= 0).all():
            errors.append(f"{c} must be >= 0")
    if TARGET in df.columns and not (df[TARGET] > 0).all():
        errors.append(f"{TARGET} must be > 0")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Run data checks on a CSV")
    parser.add_argument("csv", nargs="?", default="tests/data/sample.csv")
    args = parser.parse_args()

    errors = check_data(pd.read_csv(args.csv))
    if errors:
        print(f"Data checks FAILED for {args.csv}:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print(f"Data checks passed for {args.csv}")


if __name__ == "__main__":
    main()
