# Reproducible Prepare Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a reproducible DVC `prepare` stage that loads raw data, cleans it using tailored logic, and produces versioned train/test splits.

**Architecture:** Logic is separated into `src/data_loader.py` (functions) and `src/prepare.py` (orchestration). Configuration is driven by `params.yaml`.

**Tech Stack:** Python, Pandas, Scikit-learn, NumPy.

## Global Constraints
- **Seed**: All randomness must use the seed from `params.yaml`.
- **Paths**: No hardcoded absolute paths; use relative paths.
- **Reproducibility**: Running `src/prepare.py` twice must produce identical files.

---

### Task 1: Tailored Cleaning Logic

**Files:**
- Modify: `src/data_loader.py`
- Test: `tests/test_data_loader.py`

**Interfaces:**
- Produces: `clean_data(df: pd.DataFrame) -> pd.DataFrame`

- [ ] **Step 1: Write the failing test for `clean_data`**

```python
import pandas as pd
import numpy as np
from src.data_loader import clean_data

def test_clean_data_removes_metadata_and_constants():
    # Setup: df with metadata, a constant col, and skewed target
    df = pd.DataFrame({
        "url": ["a", "b"],
        "timedelta": [1.0, 2.0],
        "constant_col": [1, 1],
        "feature_1": [10, 20],
        "shares": [100, 1000]
    })
    cleaned = clean_data(df)
    
    assert "url" not in cleaned.columns
    assert "timedelta" not in cleaned.columns
    assert "constant_col" not in cleaned.columns
    assert "shares" in cleaned.columns
    # Check log1p transformation (approx)
    assert cleaned["shares"].iloc[0] == np.log1p(100)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_loader.py -v`
Expected: FAIL (ImportError or AttributeError: clean_data not defined)

- [ ] **Step 3: Implement `clean_data` in `src/data_loader.py`**

```python
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Remove metadata
    df = df.drop(columns=[c for c in NON_PREDICTIVE_COLS if c in df.columns])
    
    # 2. Remove zero-variance columns
    # ponytail: simple variance check, sufficient for this dataset size
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    df = df.drop(columns=constant_cols)
    
    # 3. Target transformation
    if "shares" in df.columns:
        df["shares"] = np.log1p(df["shares"])
        
    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_data_loader.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data_loader.py tests/test_data_loader.py
git commit -m "feat: add tailored cleaning logic to data_loader"
```

---

### Task 2: Prepare Orchestration Script

**Files:**
- Create: `src/prepare.py`
- Test: `tests/test_prepare.py`

**Interfaces:**
- Consumes: `src.data_loader.load_data`, `src.data_loader.clean_data`, `src.data_loader.split_data`
- Consumes: `src.utils.load_params`
- Produces: `data/processed/train.csv`, `data/processed/test.csv`

- [ ] **Step 1: Write the end-to-end test for `prepare.py`**

```python
import os
from pathlib import Path
import subprocess

def test_prepare_script_creates_files():
    result = subprocess.run(["python", "src/prepare.py"], capture_output=True, text=True)
    assert result.returncode == 0
    assert Path("data/processed/train.csv").exists()
    assert Path("data/processed/test.csv").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prepare.py -v`
Expected: FAIL (FileNotFoundError: src/prepare.py not found)

- [ ] **Step 3: Implement `src/prepare.py`**

```python
import pandas as pd
from pathlib import Path
from src.utils import load_params
from src.data_loader import load_data, clean_data, split_data

def main():
    params = load_params()
    
    # 1. Load
    df = load_data(params["data"]["raw_path"])
    
    # 2. Clean
    df = clean_data(df)
    
    # 3. Split
    X_train, X_test, y_train, y_test = split_data(
        df, 
        target_col=params["data"]["target_col"],
        test_size=params["data"]["test_size"],
        random_state=params["data"]["random_state"]
    )
    
    # 4. Save
    out_dir = Path(params["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_df.to_csv(out_dir / "train.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    print("Data preparation complete. Files saved to data/processed/")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prepare.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/prepare.py tests/test_prepare.py
git commit -m "feat: implement prepare orchestration script"
```
