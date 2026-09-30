from pathlib import Path

import pytest

from src.training.pipelines import train_sequence_pipeline
from src.utils.seed import set_seed

torch = pytest.importorskip("torch", reason="Install the deep extra to test sequence models.")

pytestmark = pytest.mark.deep


@pytest.mark.parametrize("model_name", ["lstm", "tcn"])
def test_sequence_training_exports_finite_metrics_and_checkpoint(demo_config, model_name):
    torch.set_num_threads(1)
    set_seed(42)
    demo_config["features"] = {
        "mode": "sequence", "sequence_columns": ["gaze_x", "gaze_y", "pupil", "validity"],
        "window_size": 32, "stride": 32,
    }
    demo_config["model"] = {
        "family": "deep", "name": model_name, "input_dim": 4, "hidden_dim": 8,
        "num_layers": 1, "dropout": 0.0, "num_classes": 2,
    }
    demo_config["training"].update(epochs=1, batch_size=8, lr=0.001, patience=1)
    metrics = train_sequence_pipeline(demo_config)
    assert metrics["best_epoch"] == 1
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert torch.isfinite(torch.tensor(metrics["best_val_loss"]))
    checkpoint = Path(demo_config["training"]["output_dir"]) / "checkpoints" / f"{model_name}.pt"
    assert checkpoint.stat().st_size > 0
