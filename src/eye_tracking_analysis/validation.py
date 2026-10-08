from __future__ import annotations

import numpy as np
import pandas as pd

IDS = ["participant_id", "session_id", "trial_id"]


def validate_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate without sorting timestamps, filling metadata, or inventing labels."""
    required = IDS + ["gaze_x", "gaze_y"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Empty dataset")
    if not frame.columns.is_unique:
        raise ValueError("Duplicate column names")
    result = frame.copy().reset_index(drop=True)
    for name in IDS:
        if result[name].isna().any() or result[name].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing identifier: {name}")
        result[name] = result[name].astype(str)
    for name in ["gaze_x", "gaze_y", "timestamp", "blink"]:
        if name not in result:
            continue
        try:
            result[name] = pd.to_numeric(result[name], errors="raise").astype(float)
        except (ValueError, TypeError) as error:
            raise ValueError(f"Non-numeric {name}") from error
        if np.isinf(result[name].to_numpy()).any():
            raise ValueError(f"Infinite {name}")
    if "blink" in result and not result["blink"].isin([0, 1]).all():
        raise ValueError("blink must contain explicit 0/1 observations without missing values")
    if "timestamp" in result:
        if result["timestamp"].isna().any():
            raise ValueError("Missing timestamp")
        for _, group in result.groupby(IDS, sort=False):
            if (np.diff(group["timestamp"].to_numpy()) <= 0).any():
                raise ValueError("Timestamps must be strictly increasing within participant/session/trial")
    return result
