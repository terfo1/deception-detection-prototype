import numpy as np
import pytest

from src.features.windowing import create_sliding_windows
from src.training.pipelines import split_by_participant


def test_windows_never_cross_trial_boundaries(eye_frame):
    frame = eye_frame.copy()
    for index, (_, group) in enumerate(frame.groupby(["participant_id", "trial_id"])):
        frame.loc[group.index, "gaze_x"] = index
    result = create_sliding_windows(frame, ["gaze_x", "gaze_y"], window_size=8, stride=4)
    assert result.X.shape == (18, 8, 2)
    assert (np.ptp(result.X[:, :, 0], axis=1) == 0).all()
    assert len(set(zip(result.participant_ids, result.trial_ids))) == 6


def test_short_trials_produce_empty_dataset(eye_frame):
    result = create_sliding_windows(eye_frame, ["gaze_x"], window_size=100, stride=10)
    assert result.X.shape == (0, 100, 1)


@pytest.mark.parametrize("window,stride", [(0, 1), (8, 0), (-1, 2)])
def test_invalid_window_settings_are_rejected(eye_frame, window, stride):
    with pytest.raises(ValueError, match="positive"):
        create_sliding_windows(eye_frame, ["gaze_x"], window, stride)


def test_split_has_no_participant_leakage_and_is_repeatable():
    participants = np.repeat([f"p{i:02d}" for i in range(12)], 3)
    splits = split_by_participant(participants, 0.7, 0.15, 42)
    groups = [set(participants[index]) for index in splits]
    assert all(groups)
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])
    np.testing.assert_array_equal(np.sort(np.concatenate(splits)), np.arange(len(participants)))
    for actual, repeated in zip(splits, split_by_participant(participants, 0.7, 0.15, 42)):
        np.testing.assert_array_equal(actual, repeated)
