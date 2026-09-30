from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

NUMERIC_COLUMNS = ["timestamp", "gaze_x", "gaze_y", "pupil", "blink", "validity", "confidence", "label"]


def _ensure_metadata_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Restore required metadata columns after dataframe transforms."""
    if any(name in {"participant_id", "trial_id"} for name in frame.index.names):
        frame = frame.reset_index()

    frame = frame.copy()
    if "participant_id" not in frame.columns and "source_file" in frame.columns:
        LOGGER.warning("participant_id missing after preprocessing; deriving it from source_file.")
        frame["participant_id"] = (
            frame["source_file"].astype("string").fillna("unknown").str.replace(r"\.[^.]+$", "", regex=True)
        )

    if "trial_id" not in frame.columns:
        LOGGER.warning("trial_id missing after preprocessing; defaulting all rows to 0.")
        frame["trial_id"] = 0

    return frame


def _coerce_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def sort_by_timestamp(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort records within participant and trial by timestamp when available."""
    if frame["timestamp"].isna().all():
        return frame.reset_index(drop=True)
    return frame.sort_values(["participant_id", "trial_id", "timestamp"]).reset_index(drop=True)


def filter_validity(frame: pd.DataFrame, minimum: float | None) -> pd.DataFrame:
    """Filter rows using a validity threshold when such a column exists."""
    if minimum is None or frame["validity"].isna().all():
        return frame
    filtered = frame.loc[frame["validity"].fillna(0.0) >= minimum].copy()
    LOGGER.info("Validity filter kept %s / %s rows.", len(filtered), len(frame))
    return filtered


def interpolate_missing(frame: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    """Interpolate gaze-like numeric columns within each trial."""
    frame = frame.copy()
    for column in ["gaze_x", "gaze_y", "pupil", "confidence", "validity"]:
        if frame[column].notna().any():
            frame[column] = frame.groupby(["participant_id", "trial_id"])[column].transform(
                lambda series: series.interpolate(limit=limit, limit_direction="both")
            )
    return frame


def smooth_offline(frame: pd.DataFrame, method: str | None, window: int = 5) -> pd.DataFrame:
    """Apply future-looking smoothing suitable for offline analysis."""
    if method is None:
        return frame
    frame = frame.copy()
    for column in ["gaze_x", "gaze_y", "pupil"]:
        if frame[column].notna().any() and method == "rolling_mean":
            frame[column] = frame.groupby(["participant_id", "trial_id"])[column].transform(
                lambda series: series.rolling(window=window, min_periods=1, center=True).mean()
            )
    return frame


def smooth_online_safe(frame: pd.DataFrame, method: str | None, alpha: float = 0.3) -> pd.DataFrame:
    """Apply causal smoothing that does not use future samples."""
    if method is None:
        return frame
    frame = frame.copy()
    for column in ["gaze_x", "gaze_y", "pupil"]:
        if frame[column].notna().any() and method == "ewm":
            frame[column] = frame.groupby(["participant_id", "trial_id"])[column].transform(
                lambda series: series.ewm(alpha=alpha, adjust=False).mean()
            )
    return frame


def standardize(frame: pd.DataFrame, kind: str | None) -> pd.DataFrame:
    """Standardize continuous columns globally, per participant, or per trial."""
    if kind is None:
        return frame

    def _zscore(series: pd.Series) -> pd.Series:
        std = series.std(ddof=0)
        if pd.isna(std) or std == 0:
            return series - series.mean()
        return (series - series.mean()) / std

    group_keys = None
    if kind == "participant":
        group_keys = ["participant_id"]
    elif kind == "trial":
        group_keys = ["participant_id", "trial_id"]
    elif kind != "global":
        raise ValueError(f"Unsupported normalization kind: {kind}")

    frame = frame.copy()
    for column in ["gaze_x", "gaze_y", "pupil"]:
        if frame[column].notna().any():
            if group_keys is None:
                frame[column] = _zscore(frame[column])
            else:
                frame[column] = frame.groupby(group_keys)[column].transform(_zscore)
    return frame


def normalize_pupil_baseline(frame: pd.DataFrame, fraction: float = 0.1) -> pd.DataFrame:
    """Normalize pupil values against a trial-initial baseline when possible."""
    if not 0 < fraction <= 1:
        raise ValueError("Pupil baseline fraction must be in (0, 1].")
    if frame["pupil"].isna().all():
        return frame

    frame = frame.copy()

    def _normalize(pupil: pd.Series) -> pd.Series:
        baseline_count = max(1, int(len(pupil) * fraction))
        baseline = pupil.iloc[:baseline_count].mean()
        if pd.isna(baseline) or baseline <= 0:
            return pupil
        return (pupil - baseline) / baseline

    # A Series transform preserves every metadata column on pandas 2 and 3.
    frame["pupil"] = frame.groupby(["participant_id", "trial_id"])["pupil"].transform(
        _normalize
    )
    return frame


def preprocess_eye_tracking(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Run the configured preprocessing pipeline."""
    frame = _coerce_numeric(frame)
    if config.get("sort_by_timestamp", True):
        frame = sort_by_timestamp(frame)
    frame = filter_validity(frame, config.get("validity_min"))
    if config.get("interpolate", False):
        frame = interpolate_missing(frame, limit=int(config.get("interpolation_limit", 5)))

    smoothing = config.get("smoothing", {})
    mode = config.get("mode", "offline")
    if mode == "offline":
        frame = smooth_offline(frame, method=smoothing.get("method"), window=int(smoothing.get("window", 5)))
    elif mode == "online_safe":
        frame = smooth_online_safe(frame, method=smoothing.get("method"), alpha=float(smoothing.get("alpha", 0.3)))
    else:
        raise ValueError(f"Unsupported preprocessing mode: {mode}")

    pupil_cfg = config.get("pupil_baseline", {})
    if pupil_cfg.get("enabled", False):
        frame = normalize_pupil_baseline(frame, fraction=float(pupil_cfg.get("fraction", 0.1)))

    # Relative pupil change must be calculated from physical diameters,
    # before z-scoring can introduce negative or near-zero baselines.
    normalization = config.get("normalization", {})
    frame = standardize(frame, kind=normalization.get("kind"))

    frame = _ensure_metadata_columns(frame)
    return frame.reset_index(drop=True)
