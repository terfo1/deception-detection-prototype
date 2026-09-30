import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.training.pipelines import train_baseline_pipeline


@pytest.mark.parametrize("model_name", ["logistic_regression", "random_forest", "svm"])
def test_baseline_pipeline_exports_usable_artifacts(demo_config, model_name):
    demo_config["model"]["name"] = model_name
    metrics = train_baseline_pipeline(demo_config)
    assert metrics["n_train"] + metrics["n_val"] + metrics["n_test"] == 48
    assert 0.0 <= metrics["accuracy"] <= 1.0
    out = Path(demo_config["training"]["output_dir"])
    saved = json.loads((out / "metrics" / f"{model_name}_metrics.json").read_text())
    assert saved == metrics
    assert (out / "figures" / f"{model_name}_confusion_matrix.png").stat().st_size > 0
    assert (out / "figures" / f"{model_name}_roc_curve.png").stat().st_size > 0
    with (out / "checkpoints" / f"{model_name}.pkl").open("rb") as handle:
        checkpoint = pickle.load(handle)
    probabilities = checkpoint["model"].predict_proba(np.zeros((1, len(checkpoint["feature_columns"]))))
    assert probabilities.shape == (1, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), [1.0])


def test_documented_cli_runs_without_external_data(demo_config, tmp_path):
    config_path = tmp_path / "demo.yaml"
    config_path.write_text(yaml.safe_dump(demo_config), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "scripts/train.py", "--config", str(config_path)],
        capture_output=True, text=True, check=True, timeout=90,
    )
    metrics = json.loads(result.stdout)
    assert metrics["n_test"] > 0
