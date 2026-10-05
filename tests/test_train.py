import yaml

from src.train import train_pipeline


def test_train_pipeline_smoke(tmp_path):
    """Smoke test baseline train pipeline with synthetic data fallback using tmp_path."""
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    temp_metrics = tmp_path / "smoke_metrics.json"
    params_copy = dict(params)
    params_copy["evaluate"] = dict(params_copy.get("evaluate", {}))
    params_copy["evaluate"]["metrics_file"] = str(temp_metrics)

    temp_params_path = tmp_path / "params.yaml"
    with open(temp_params_path, "w", encoding="utf-8") as f:
        yaml.dump(params_copy, f)

    metrics = train_pipeline(config_path=str(temp_params_path))

    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert metrics["rmse"] >= 0.0
    assert temp_metrics.exists()
