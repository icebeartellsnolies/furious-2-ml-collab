# Design: Reproducible Data Preparation Stage (Phase 6)
**Date:** 2026-09-29
**Role:** Data Owner
**Status:** Draft

## 1. Purpose
Transform the raw `OnlineNewsPopularity` dataset into versioned, cleaned, and deterministically split training and testing sets. This ensures that every team member starts their model training from the exact same data state.

## 2. Architecture
The process is encapsulated in a single DVC stage called `prepare`.

**Pipeline:** 
`data/raw/dataset.csv` $\rightarrow$ `src/prepare.py` $\rightarrow$ `data/processed/train.csv` & `data/processed/test.csv`

## 3. Component Specifications

### 3.1 Logic Layer (`src/data_loader.py`)
Existing functions `load_data` and `split_data` will be augmented with a new cleaning function.

- **`load_data()`**: 
    - Loads raw CSV.
    - Strips whitespace from column names.
    - Handles synthetic fallback for CI.
- **`clean_data(df)`**: (New)
    - **Metadata Removal**: Drops `url` and `timedelta`.
    - **Zero-Variance Filter**: Identifies and drops columns where all values are identical (constant features).
    - **Target Transformation**: Applies `np.log1p` to the `shares` column to normalize the heavily right-skewed distribution.
- **`split_data()`**:
    - Uses `train_test_split` with `random_state` from `params.yaml`.
    - Returns feature matrices and target vectors.

### 3.2 Orchestration Layer (`src/prepare.py`)
A new script that acts as the DVC entry point:
1. Calls `utils.load_params()` to get `seed`, `test_size`, and `target_col`.
2. Executes the sequence: `load_data` $\rightarrow$ `clean_data` $\rightarrow$ `split_data`.
3. Merges `X_train` and `y_train` $\rightarrow$ saves to `data/processed/train.csv`.
4. Merges `X_test` and `y_test` $\rightarrow$ saves to `data/processed/test.csv`.

## 4. Reproducibility Guarantee
- **Config-Driven**: All magic numbers (seed, split ratio) live in `params.yaml`.
- **Deterministic**: The same raw data and seed will always produce the same `train.csv` and `test.csv` hashes.
- **DVC Tracked**: The output files will be tracked by DVC, allowing teammates to sync via `dvc pull`.

## 5. Success Criteria
- `python src/prepare.py` runs without errors.
- `data/processed/train.csv` and `test.csv` are created.
- Running the script twice produces identical files (verified by hash).
