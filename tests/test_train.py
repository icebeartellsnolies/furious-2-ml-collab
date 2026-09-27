from pathlib import Path

from src.train import train_pipeline


def test_train_pipeline_smoke(tmp_path):
    # Train pipeline with synthetic data fallback
    metrics = train_pipeline(config_path="params.yaml")

    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert metrics["rmse"] >= 0.0

    # Verify model artifact and metrics file were produced
    assert Path("models/baseline_model.joblib").exists()
    assert Path("metrics.json").exists()
