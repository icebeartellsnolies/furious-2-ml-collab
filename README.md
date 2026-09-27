# Furious-2 ML Collaboration: Online News Popularity

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Dependency Manager](https://img.shields.io/badge/managed%20by-uv-purple.svg)](https://github.com/astral-sh/uv)
[![Code Style](https://img.shields.io/badge/code%20style-ruff-black.svg)](https://github.com/astral-sh/ruff)

An end-to-end reproducible MLOps project for predicting online news article popularity (social shares) using collaborative Git workflows, DVC data versioning, automated testing, and CI/CD pipelines.

---

## 1. Project Overview

- **Dataset:** [Online News Popularity](https://www.kaggle.com/datasets/deepakshende/onlinenewspopularity) (UCI Machine Learning Repository)
- **Task:** Predict the number of social media shares (Regression & Popularity Classification)
- **Goal:** Strict end-to-end reproducibility across team members: *Every result, reproducible by anyone on the team.*

### Team Furious-2 Roles
- **Data Owner:** Bisma Munir (`@icebeartellsnolies`) - DVC management, data validation, dataset updates, split management.
- **Model Owner:** Model Owner - Model architecture, training pipeline, hyperparameters (`params.yaml`), experiment tracking (`dvc exp`).
- **Platform Owner:** Shared - Project scaffolding, dependency management (`uv`), CI/CD workflows, release lifecycle (`model-v1.0`).

---

## 2. Repository Layout

```text
.
├── configs/              # Hyperparameter & configuration files (params.yaml)
├── data/                 # Data directory (managed by DVC, ignored by Git)
│   ├── raw/              # Raw OnlineNewsPopularity.csv
│   └── processed/        # Preprocessed train/test splits
├── models/               # Serialized model artifacts (tracked by DVC)
├── notebooks/            # Jupyter notebooks paired with Jupytext
├── src/                  # Modular, testable pipeline source code
│   ├── data_loader.py    # Robust data ingestion and splitting
│   ├── train.py          # Baseline model training and evaluation
│   └── utils.py          # Reproducibility and helper functions
├── tests/                # Automated pytest unit & smoke tests
├── .github/              # PR templates and GitHub Actions CI workflows
│   ├── workflows/ci.yml  # Lint, tests, data checks, smoke train
│   └── pull_request_template.md
├── CONTRIBUTING.md       # Branching, commits, and PR checklist guidelines
├── pyproject.toml        # Project metadata and dependencies
├── uv.lock               # Deterministic dependency lockfile
└── README.md
```

---

## 3. Quickstart

### Prerequisites
- Install [uv](https://docs.astral.sh/uv/) (fast Python package manager)
- Git 2.30+

### Setup Environment
```bash
# 1. Clone repository
git clone https://github.com/icebeartellsnolies/furious-2-ml-collab.git
cd furious-2-ml-collab

# 2. Synchronize virtual environment with pinned lockfile
uv sync

# 3. Pull datasets and models with DVC (requires remote access)
dvc pull
```

### Running the Baseline Model
```bash
# Run baseline training pipeline from command line
uv run python src/train.py
```

### Running Tests and Linters
```bash
# Run test suite
uv run pytest tests/

# Run linter & formatter checks
uv run ruff check .
uv run ruff format --check .
```
