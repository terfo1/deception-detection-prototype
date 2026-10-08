import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eye_tracking_analysis import AnalysisConfig, analyze, export_results, load_csv


def gaze(x=(0, 0, 3, 3), y=(0, 0, 4, 4), t=(0, 0.1, 0.2, 0.3)):
    return pd.DataFrame({"participant_id": "001", "session_id": "01", "trial_id": "01",
                         "gaze_x": x, "gaze_y": y, "timestamp": t})


def config(**kwargs):
    return AnalysisConfig("degrees", 30, "s", max_gap_s=0.2, min_fixation_s=0.09, **kwargs)


def test_csv_preserves_identifiers_and_checksum(tmp_path):
    path = tmp_path / "gaze.csv"
    gaze().to_csv(path, index=False)
    loaded = load_csv(path)
    assert loaded["participant_id"].tolist() == ["001"] * 4
    assert loaded["session_id"].tolist() == ["01"] * 4
    assert len(loaded.attrs["input_sha256"]) == 64
    assert "label" not in loaded


@pytest.mark.parametrize("text,match", [("", "Empty"), ("gaze_x,gaze_y\n", "Missing|Empty"),
                                         ("gaze_x,gaze_x\n1,1", "Duplicate")])
def test_invalid_csv(tmp_path, text, match):
    path = tmp_path / "bad.csv"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_csv(path)


@pytest.mark.parametrize("column,value,match", [("gaze_x", "bad", "Non-numeric"),
                                                 ("gaze_x", np.inf, "Infinite"),
                                                 ("participant_id", " ", "identifier"),
                                                 ("session_id", None, "identifier"),
                                                 ("timestamp", np.nan, "timestamp"),
                                                 ("blink", 2, "blink")])
def test_invalid_observations(column, value, match):
    frame = gaze().astype(object)
    frame.loc[1, column] = value
    with pytest.raises(ValueError, match=match):
        analyze(frame, config())


@pytest.mark.parametrize("times", [(0, 0.1, 0.1, 0.2), (0, 0.2, 0.1, 0.3)])
def test_timestamps_rejected_without_sorting(times):
    with pytest.raises(ValueError, match="strictly increasing"):
        analyze(gaze(t=times), config())


def test_empty_frame():
    with pytest.raises(ValueError, match="Empty"):
        analyze(gaze().iloc[:0], config())


def test_known_345_saccade_fixations_and_dispersion():
    result = analyze(gaze(), config())
    feature = result.features.iloc[0]
    assert feature["fixation_count"] == 2
    assert feature["mean_fixation_duration_s"] == pytest.approx(0.1)
    assert feature["saccade_count"] == 1
    assert feature["mean_saccade_amplitude"] == pytest.approx(5)
    assert feature["mean_saccade_peak_velocity"] == pytest.approx(50)
    assert feature["gaze_dispersion"] == pytest.approx(2.5)
    event = result.events.query("event_type == 'saccade'").iloc[0]
    assert event["duration_s"] == pytest.approx(0.1)
    assert event["mean_velocity"] == pytest.approx(50)
    assert pd.isna(feature["blink_rate_per_min"])


def test_milliseconds_are_equivalent_and_units_are_mandatory():
    frame = gaze()
    frame["timestamp"] *= 1000
    ms = analyze(frame, AnalysisConfig("degrees", 30, "ms", max_gap_s=0.2, min_fixation_s=0.09))
    pd.testing.assert_frame_equal(ms.events, analyze(gaze(), config()).events)
    with pytest.raises(ValueError, match="timestamp_unit"):
        analyze(frame, AnalysisConfig("degrees", 30))


def test_no_timestamps_only_spatial_features():
    frame = gaze().drop(columns="timestamp")
    result = analyze(frame, AnalysisConfig("degrees", 30))
    assert result.events.empty
    assert result.features.iloc[0]["gaze_dispersion"] == pytest.approx(2.5)
    assert pd.isna(result.features.iloc[0]["duration_s"])
    assert pd.isna(result.features.iloc[0]["saccade_count"])
    with pytest.raises(ValueError, match="requires timestamps"):
        analyze(frame, AnalysisConfig("degrees", 30, smoothing_ms=20))


def test_screen_geometry_normalization_without_fitting():
    frame = gaze(x=(0, 100, 200, 300), y=(0, 50, 100, 150))
    before = frame.copy()
    result = analyze(frame, AnalysisConfig("pixels", 3, "s", max_gap_s=0.2, screen_size=(1000, 500)))
    np.testing.assert_allclose(result.samples["gaze_x"], [0, 0.1, 0.2, 0.3])
    np.testing.assert_allclose(result.samples["gaze_y"], [0, 0.1, 0.2, 0.3])
    assert result.metadata["coordinate_unit"] == "normalized"
    pd.testing.assert_frame_equal(frame, before)


def test_bounded_interpolation_uses_time_and_preserves_edges():
    frame = gaze(x=(np.nan, 0, np.nan, 3, np.nan), y=(np.nan, 0, np.nan, 0, np.nan), t=(0, 0.1, 0.15, 0.3, 0.4))
    result = analyze(frame, config(interpolate_max_s=0.21))
    assert result.samples.loc[2, "gaze_x"] == pytest.approx(0.75)
    assert result.samples.loc[[0, 4], "gaze_x"].isna().all()
    assert result.features.iloc[0]["interpolated_samples"] == 1
    assert result.features.iloc[0]["missing_input_fraction"] == pytest.approx(3 / 5)
    unfilled = analyze(frame, config(interpolate_max_s=0.19))
    assert pd.isna(unfilled.samples.loc[2, "gaze_x"])


def test_long_missing_run_not_partially_filled():
    frame = gaze(x=(0, np.nan, np.nan, 3), y=(0, np.nan, np.nan, 4))
    result = analyze(frame, config(interpolate_max_s=0.2))
    assert result.samples["gaze_usable"].tolist() == [True, False, False, True]
    assert result.events.empty


@pytest.mark.parametrize("changed_id", ["participant_id", "session_id", "trial_id"])
def test_no_cross_boundary_events_interpolation_or_smoothing(changed_id):
    first = gaze(x=(0, 0), y=(0, 0), t=(0, 0.1))
    second = gaze(x=(np.nan, 100), y=(np.nan, 0), t=(0, 0.1))
    second[changed_id] = "different"
    result = analyze(pd.concat([first, second], ignore_index=True), config(interpolate_max_s=1, smoothing_ms=500))
    assert len(result.features) == 2
    assert result.samples["gaze_x"].iloc[:2].tolist() == [0, 0]
    assert pd.isna(result.samples["gaze_x"].iloc[2])
    assert result.samples["gaze_x"].iloc[3] == 100
    assert (result.events["event_type"] == "fixation").all()


def test_acquisition_gap_not_a_saccade_or_smoothing_bridge():
    frame = gaze(t=(0, 0.1, 1.1, 1.2))
    result = analyze(frame, config(smoothing_ms=2000))
    feature = result.features.iloc[0]
    assert feature["timestamp_gap_count"] == 1
    assert feature["observed_duration_s"] == pytest.approx(0.2)
    assert feature["saccade_count"] == 0
    assert result.samples["gaze_x"].tolist() == [0, 0, 3, 3]


def test_time_based_filter_uses_recorded_frequency_and_never_fills_missing():
    slow = analyze(gaze(x=(0, 2, 4, 6), y=(0, 0, 0, 0)), config(smoothing_ms=150))
    fast = analyze(gaze(x=(0, 2, 4, 6), y=(0, 0, 0, 0), t=(0, 0.05, 0.1, 0.15)), config(smoothing_ms=150))
    assert slow.samples["gaze_x"].iloc[2] == pytest.approx(3)
    assert fast.samples["gaze_x"].iloc[2] == pytest.approx(2)
    assert slow.features["sampling_rate_hz_estimate"].iloc[0] == pytest.approx(10)
    assert fast.features["sampling_rate_hz_estimate"].iloc[0] == pytest.approx(20)
    missing = analyze(gaze(x=(0, np.nan, 4, 6)), config(smoothing_ms=500))
    assert pd.isna(missing.samples.loc[1, "gaze_x"])
    assert missing.samples.loc[2, "gaze_x"] == 4


def test_blink_rate_requires_explicit_flags_and_masks_blink_gaze():
    frame = gaze()
    frame["blink"] = [0, 1, 1, 0]
    result = analyze(frame, config(interpolate_max_s=1))
    assert result.features.iloc[0]["blink_onset_count"] == 1
    assert result.features.iloc[0]["blink_rate_per_min"] == pytest.approx(200)
    assert result.samples.loc[[1, 2], "gaze_x"].isna().all()
    assert result.features.iloc[0]["interpolated_samples"] == 0
    frame["blink"] = [1, 1, 0, 0]
    assert analyze(frame, config()).features.iloc[0]["blink_onset_count"] == 0


def test_all_missing_is_unavailable_not_zero_dispersion():
    result = analyze(gaze(x=(np.nan,) * 4), config())
    assert pd.isna(result.features.iloc[0]["gaze_dispersion"])
    assert result.features.iloc[0]["usable_samples"] == 0
    assert result.events.empty


def test_labels_do_not_affect_analysis():
    frame = gaze()
    frame["label"] = ["unknown", None, "CIT", "lie"]
    result = analyze(frame, config())
    pd.testing.assert_frame_equal(result.features, analyze(gaze(), config()).features)
    assert result.metadata["labels_used"] is False


def test_export_values_headers_provenance_and_overwrite_refusal(tmp_path):
    result = analyze(gaze(), config())
    output = export_results(result, tmp_path / "new-run")
    features = pd.read_csv(output / "features.csv")
    assert features.loc[0, "mean_saccade_amplitude"] == pytest.approx(5)
    assert pd.read_csv(output / "events.csv").shape[0] == 3
    metadata = json.loads((output / "analysis.json").read_text())
    assert metadata["config"]["velocity_threshold"] == 30
    assert metadata["module_sha256"]["analysis.py"]
    assert (output / "gaze_analysis.png").stat().st_size > 1000
    with pytest.raises(FileExistsError):
        export_results(result, output)


@pytest.mark.parametrize("kwargs", [{"coordinate_unit": "unknown"}, {"velocity_threshold": 0},
                                    {"max_gap_s": np.nan}, {"smoothing_ms": -1},
                                    {"screen_size": (100, 100)}, {"timestamp_unit": "guess"}])
def test_invalid_configuration(kwargs):
    values = {"coordinate_unit": "degrees", "velocity_threshold": 30, **kwargs}
    with pytest.raises(ValueError):
        AnalysisConfig(**values)


def test_documented_demo_and_cli(tmp_path):
    root = Path(__file__).resolve().parents[1]
    demo = tmp_path / "demo"
    process = subprocess.run([sys.executable, str(root / "examples/eye_tracking_analysis_demo.py"),
                              "--output-dir", str(demo)], cwd=root, text=True, capture_output=True)
    assert process.returncode == 0, process.stderr
    assert "SYNTHETIC DATA" in process.stdout
    feature = pd.read_csv(demo / "analysis/features.csv").iloc[0]
    assert feature["fixation_count"] == 2
    assert feature["mean_saccade_amplitude"] == pytest.approx(5)
    assert feature["mean_saccade_peak_velocity"] == pytest.approx(250)
    cli = subprocess.run([sys.executable, "-m", "src.eye_tracking_analysis.cli", str(demo / "synthetic_gaze.csv"),
                          "--output-dir", str(tmp_path / "cli"), "--coordinate-unit", "degrees",
                          "--timestamp-unit", "ms", "--velocity-threshold", "30"],
                         cwd=root, text=True, capture_output=True)
    assert cli.returncode == 0, cli.stderr
    assert pd.read_csv(tmp_path / "cli/features.csv").loc[0, "saccade_count"] == 1
