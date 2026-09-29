import pandas as pd
from pathlib import Path
from src.utils import load_params
from src.data_loader import load_data, clean_data, split_data


def main():
    # Load configuration
    params = load_params()

    # 1. Load raw data
    # load_data handles the fallback to synthetic data for CI/smoke tests
    df = load_data(params["data"]["raw_path"])

    # 2. Clean data
    # Removes metadata, constant columns, and log-transforms the target
    df = clean_data(df)

    # 3. Split into training and testing sets
    X_train, X_test, y_train, y_test = split_data(
        df,
        target_col=params["data"]["target_col"],
        test_size=params["data"]["test_size"],
        random_state=params["data"]["random_state"],
    )

    # 4. Save processed files
    out_dir = Path(params["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Merge features and target for CSV storage
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(out_dir / "train.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    print(f"Data preparation complete. Files saved to: {out_dir}")


if __name__ == "__main__":
    main()
