from pathlib import Path

import numpy as np
import pandas as pd

from src.data.adapters import CITAdapter
from src.evaluation.metrics import compute_classification_metrics
from src.inference.streaming import SlidingWindowInferenceAggregator


def test_cit_excel_maps_groups_and_separates_stimuli(tmp_path):
    raw = pd.DataFrame({
        "Participant name": ["p1"] * 4,
        "group": ["guilty"] * 4,
        "Recording timestamp": [0, 10, 20, 30],
        "Gaze point X": [100, 110, 120, 130],
        "Gaze point Y": [200, 210, 220, 230],
        "Pupil diameter filtered": [3.0] * 4,
        "Validity left": ["Valid", "Invalid", "Valid", "Valid"],
        "Presented Stimulus name": ["a", "a", "b", "b"],
    })
    source = Path(tmp_path) / "recording.xlsx"
    raw.to_excel(source, index=False)
    frame = CITAdapter().load_file(source)
    assert frame["label"].tolist() == [1] * 4
    assert frame["trial_id"].tolist() == [0, 0, 1, 1]
    assert frame["validity"].tolist() == [1.0, 0.0, 1.0, 1.0]
    assert frame["participant_id"].tolist() == ["p1"] * 4


def test_binary_metrics_match_known_confusion_matrix():
    result = compute_classification_metrics(
        np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), np.array([0.1, 0.6, 0.8, 0.9])
    )
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]
    assert result["accuracy"] == 0.75
    assert result["roc_auc"] == 1.0


def test_streaming_aggregator_discards_old_predictions():
    aggregator = SlidingWindowInferenceAggregator(history_size=2)
    assert aggregator.update(0.0) == 0.0
    assert aggregator.update(1.0) == 0.5
    assert aggregator.update(1.0) == 1.0
