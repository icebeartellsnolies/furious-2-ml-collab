from pathlib import Path

import yaml

from src.evaluate import evaluate_stage
from src.prepare import prepare_stage
from src.train_stage import train_stage


def test_dvc_pipeline_stages_end_to_end(tmp_path):
    """Verify end-to-end execution of prepare, train, and evaluate pipeline stages writing output to tmp_path."""
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    temp_metrics = tmp_path / "pipeline_metrics.json"
    params_copy = dict(params)
    params_copy["evaluate"] = dict(params_copy.get("evaluate", {}))
    params_copy["evaluate"]["metrics_file"] = str(temp_metrics)

    temp_params_path = tmp_path / "params.yaml"
    with open(temp_params_path, "w", encoding="utf-8") as f:
        yaml.dump(params_copy, f)

    prepare_stage(config_path=str(temp_params_path))
    assert Path("data/processed/train.csv").exists()
    assert Path("data/processed/test.csv").exists()

    train_stage(config_path=str(temp_params_path))
    assert Path("models/model.joblib").exists()

    metrics = evaluate_stage(config_path=str(temp_params_path))
    assert temp_metrics.exists()
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert "commit_sha" in metrics
    assert "hyperparameters" in metrics
