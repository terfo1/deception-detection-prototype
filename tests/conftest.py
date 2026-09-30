import pytest

from src.data.synthetic import make_synthetic_eye_tracking_dataset
from src.utils.config import load_config


@pytest.fixture
def eye_frame():
    return make_synthetic_eye_tracking_dataset(
        n_participants=2, trials_per_participant=3, samples_per_trial=16, seed=42
    )


@pytest.fixture
def demo_config(tmp_path):
    config = load_config("configs/synthetic_baseline.yaml")
    config["dataset"].update(trials_per_participant=2, samples_per_trial=32)
    config["training"]["output_dir"] = str(tmp_path / "outputs")
    return config
