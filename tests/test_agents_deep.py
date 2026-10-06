from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.agents.orchestrator import Orchestrator
from src.agents.schemas import SessionRequest, WorkflowConfig, WorkflowStatus
from src.agents.services import FileEyeTrackingSource, ModelSpec
from src.data.synthetic import make_synthetic_eye_tracking_dataset
from src.models.sequence import build_sequence_model
from src.training.pipelines import train_sequence_pipeline
from src.utils.seed import set_seed

torch = pytest.importorskip("torch", reason="Sequence workflow needs the existing optional deep extra.")
pytestmark = pytest.mark.deep


@pytest.mark.parametrize("name", ["lstm", "tcn"])
def test_existing_sequence_checkpoint_runs_without_labels_or_training(demo_config, tmp_path, name):
    torch.set_num_threads(1)
    set_seed(42)
    channels = ("gaze_x", "gaze_y", "pupil", "validity")
    demo_config["features"] = {"mode": "sequence", "sequence_columns": list(channels),
                               "window_size": 32, "stride": 32}
    architecture = {"name": name, "input_dim": 4, "hidden_dim": 8,
                    "num_layers": 1, "dropout": 0.0, "num_classes": 2}
    demo_config["model"] = {"family": "deep", **architecture}
    demo_config["training"].update(epochs=1, batch_size=8, lr=0.001, patience=1)
    train_sequence_pipeline(demo_config)
    checkpoint = Path(demo_config["training"]["output_dir"]) / "checkpoints" / f"{name}.pt"
    spec = ModelSpec("sequence", str(checkpoint), hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                     f"synthetic-{name}", "synthetic_debug", ((0, "debug 0"), (1, "debug 1")), "ms",
                     demo_config["preprocessing"], channels, 32, architecture)
    raw = make_synthetic_eye_tracking_dataset(1, 1, 32).drop(columns="label")
    source_path = tmp_path / "window.csv"
    raw.to_csv(source_path, index=False)
    request = SessionRequest("debug", "p00", "p00_t00", "ms")
    state = Orchestrator(FileEyeTrackingSource(source_path), spec, WorkflowConfig()).run(request)
    assert state.status == WorkflowStatus.COMPLETED
    model = build_sequence_model(architecture)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    with torch.inference_mode():
        expected = torch.softmax(model(torch.tensor(state.features.values, dtype=torch.float32).unsqueeze(0)), dim=1)
    np.testing.assert_allclose([value for _, value in state.prediction.probabilities], expected[0].numpy())
    assert state.prediction.probability_kind == "softmax_uncalibrated"
    # A full trial must not be silently shortened to fit a model trained on a different window size.
    bad = Orchestrator(FileEyeTrackingSource(source_path), replace(spec, window_size=16), WorkflowConfig()).run(request)
    assert bad.status == WorkflowStatus.FAILED
    assert bad.report.errors[-1].code == "SEQUENCE_WINDOW_SIZE_MISMATCH"
