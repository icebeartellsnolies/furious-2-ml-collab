"""Smoke test: run prepare -> train -> evaluate on the committed sample, in a temp dir.

Runs in a temp working directory so it never touches the real data/processed,
models/ or metrics.json that DVC tracks.
"""

import os
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main(sample: Path = ROOT / "tests" / "data" / "sample.csv") -> dict:
    # imported here (not at top) so the sys.path line above runs first
    from src.evaluate import evaluate_stage
    from src.prepare import prepare_stage
    from src.train_stage import train_stage

    params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    params["train"]["n_estimators"] = 10  # keep CI fast

    start = Path.cwd()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        params["data"]["raw_path"] = str(sample)
        params["data"]["processed_dir"] = str(tmp / "processed")
        params["evaluate"]["metrics_file"] = str(tmp / "metrics.json")
        cfg = tmp / "params.yaml"
        cfg.write_text(yaml.dump(params), encoding="utf-8")

        os.chdir(tmp)  # train_stage/evaluate write to ./models
        try:
            prepare_stage(str(cfg))
            train_stage(str(cfg))
            metrics = evaluate_stage(str(cfg))
        finally:
            os.chdir(start)  # Windows can't delete the cwd

    assert {"mae", "rmse", "r2"} <= metrics.keys(), metrics
    print(f"Smoke train OK: {metrics}")
    return metrics


if __name__ == "__main__":
    main()
