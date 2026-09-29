from pathlib import Path

from src.evaluate import evaluate_stage
from src.prepare import prepare_stage
from src.train_stage import train_stage


def test_dvc_pipeline_stages_end_to_end():
    """Verify end-to-end execution of prepare, train, and evaluate pipeline stages."""
    prepare_stage(config_path="params.yaml")
    assert Path("data/processed/train.csv").exists()
    assert Path("data/processed/test.csv").exists()

    train_stage(config_path="params.yaml")
    assert Path("models/model.joblib").exists()

    metrics = evaluate_stage(config_path="params.yaml")
    assert Path("metrics.json").exists()
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert "commit_sha" in metrics
    assert "hyperparameters" in metrics
