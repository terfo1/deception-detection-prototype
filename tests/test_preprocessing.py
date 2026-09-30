import numpy as np
import pandas as pd
import pytest

from src.features.aggregated import create_aggregated_features
from src.features.preprocessing import normalize_pupil_baseline, preprocess_eye_tracking


def test_preprocessing_preserves_participants_trials_and_feature_count(eye_frame):
    before = eye_frame[["participant_id", "trial_id"]].drop_duplicates()
    processed = preprocess_eye_tracking(
        eye_frame,
        {"pupil_baseline": {"enabled": True, "fraction": 0.25}, "normalization": {"kind": "participant"}},
    )
    after = processed[["participant_id", "trial_id"]].drop_duplicates()
    pd.testing.assert_frame_equal(before.reset_index(drop=True), after.reset_index(drop=True))
    assert len(processed) == len(eye_frame)
    assert len(create_aggregated_features(processed)) == 6
    assert np.isfinite(processed[["gaze_x", "gaze_y", "pupil"]].to_numpy()).all()


def test_pupil_baseline_is_trial_local_and_preserves_metadata(eye_frame):
    frame = eye_frame.iloc[:32].copy()
    frame["pupil"] = [2.0] * 8 + [4.0] * 8 + [4.0] * 8 + [8.0] * 8
    result = normalize_pupil_baseline(frame, fraction=0.5)
    np.testing.assert_allclose(result["pupil"], [0.0] * 8 + [1.0] * 8 + [0.0] * 8 + [1.0] * 8)
    pd.testing.assert_frame_equal(result.drop(columns="pupil"), frame.drop(columns="pupil"))
    assert frame["pupil"].iloc[0] == 2.0


def test_pupil_baseline_precedes_standardization(eye_frame):
    frame = eye_frame.iloc[:16].copy()
    frame["pupil"] = [2.0] * 8 + [4.0] * 8
    result = preprocess_eye_tracking(
        frame, {"pupil_baseline": {"enabled": True, "fraction": 0.5}, "normalization": {"kind": "participant"}}
    )
    np.testing.assert_allclose(result["pupil"], [-1.0] * 8 + [1.0] * 8)


@pytest.mark.parametrize("fraction", [0, -0.1, 1.1])
def test_invalid_baseline_fraction_is_rejected(eye_frame, fraction):
    with pytest.raises(ValueError, match="fraction"):
        normalize_pupil_baseline(eye_frame, fraction=fraction)


def test_invalid_rows_are_removed_and_interpolation_does_not_cross_trials(eye_frame):
    frame = eye_frame.copy()
    frame.loc[0, "validity"] = 0.0
    frame.loc[16, "gaze_x"] = np.nan
    result = preprocess_eye_tracking(frame, {"validity_min": 0.5, "interpolate": True})
    assert len(result) == len(frame) - 1
    assert result["gaze_x"].notna().all()
