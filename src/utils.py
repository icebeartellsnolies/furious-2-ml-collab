import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_params(params_path: str = "params.yaml") -> dict[str, Any]:
    """Load hyperparameters and pipeline configuration from YAML file."""
    path = Path(params_path)
    if not path.exists():
        # Fallback to configs/params.yaml if root params.yaml not found
        fallback = Path("configs/params.yaml")
        if fallback.exists():
            path = fallback
        else:
            raise FileNotFoundError(
                f"Configuration file not found at {params_path} or configs/params.yaml"
            )

    with open(path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    return params


def save_metrics(metrics: dict[str, Any], filepath: str = "metrics.json") -> None:
    """Save evaluation metrics and run metadata to JSON file."""
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)


def get_git_commit_sha() -> str:
    """Retrieve the current Git commit SHA, or 'unknown' if not in a Git repository."""
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        return sha
    except (subprocess.SubprocessError, OSError):
        return "unknown"
