from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_stats(values: pd.Series, prefix: str) -> dict[str, float]:
    clean = values.dropna().to_numpy(dtype=float)
    if clean.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_range": 0.0,
        }
    return {
        f"{prefix}_mean": float(clean.mean()),
        f"{prefix}_std": float(clean.std()),
        f"{prefix}_min": float(clean.min()),
        f"{prefix}_max": float(clean.max()),
        f"{prefix}_range": float(clean.max() - clean.min()),
    }


def _compute_dynamics(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    time = group["timestamp"].to_numpy(dtype=float, na_value=np.nan)
    x = group["gaze_x"].to_numpy(dtype=float, na_value=np.nan)
    y = group["gaze_y"].to_numpy(dtype=float, na_value=np.nan)
    if np.isnan(time).all():
        time = np.arange(len(group), dtype=float)
    x = np.nan_to_num(x, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)
    dt = np.diff(time, prepend=time[0])
    positive_dt = dt[dt > 0]
    fallback_dt = float(np.median(positive_dt)) if positive_dt.size else 1.0
    dt = np.where(dt > 0, dt, fallback_dt)
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    velocity = np.sqrt(dx**2 + dy**2) / dt
    dv = np.diff(velocity, prepend=velocity[0])
    acceleration = dv / dt
    velocity = np.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)
    acceleration = np.nan_to_num(acceleration, nan=0.0, posinf=0.0, neginf=0.0)
    return velocity, acceleration


def create_aggregated_features(frame: pd.DataFrame, *, include_labels: bool = True) -> pd.DataFrame:
    """Aggregate trial-level features from preprocessed gaze streams."""
    required = {"participant_id", "trial_id", "stimulus_id"}
    if include_labels:
        required.add("label")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Aggregated feature generation requires columns {missing}, got {list(frame.columns)}")

    records: list[dict[str, float | int | str]] = []
    grouped = frame.groupby(["participant_id", "trial_id"], dropna=False)
    for (participant_id, trial_id), group in grouped:
        velocity, acceleration = _compute_dynamics(group)
        record: dict[str, float | int | str] = {
            "participant_id": participant_id,
            "trial_id": trial_id,
        }
        if include_labels:
            record["label"] = int(group["label"].mode(dropna=True).iloc[0]) if group["label"].notna().any() else 0
        record.update({
            "stimulus_id": group["stimulus_id"].dropna().iloc[0] if group["stimulus_id"].notna().any() else "unknown",
            "n_samples": int(len(group)),
            "trial_duration": float(group["timestamp"].max() - group["timestamp"].min())
            if group["timestamp"].notna().any()
            else float(len(group)),
            "blink_count": int(group["blink"].fillna(0).astype(int).diff().fillna(0).clip(lower=0).sum())
            if group["blink"].notna().any()
            else 0,
            "fixation_count_estimate": int(
                (np.sqrt(np.diff(np.nan_to_num(group["gaze_x"]), prepend=0.0) ** 2 + np.diff(np.nan_to_num(group["gaze_y"]), prepend=0.0) ** 2) < 0.02).sum()
            ),
        })
        record.update(_safe_stats(group["gaze_x"], "gaze_x"))
        record.update(_safe_stats(group["gaze_y"], "gaze_y"))
        record.update(_safe_stats(group["pupil"], "pupil"))
        record.update(_safe_stats(pd.Series(velocity), "velocity"))
        record.update(_safe_stats(pd.Series(acceleration), "acceleration"))
        dispersion = np.sqrt(
            np.nan_to_num(group["gaze_x"].std(ddof=0), nan=0.0) ** 2
            + np.nan_to_num(group["gaze_y"].std(ddof=0), nan=0.0) ** 2
        )
        record["dispersion"] = float(dispersion)
        records.append(record)
    return pd.DataFrame.from_records(records)
