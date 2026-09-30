from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class WindowedDataset:
    """Container for sequence windows and associated metadata."""

    X: np.ndarray
    y: np.ndarray
    participant_ids: np.ndarray
    trial_ids: np.ndarray


def create_sliding_windows(
    frame: pd.DataFrame,
    sequence_columns: list[str],
    window_size: int,
    stride: int,
) -> WindowedDataset:
    """Create weakly labeled sliding windows from trial streams."""
    if window_size <= 0 or stride <= 0:
        raise ValueError("Window size and stride must be positive.")
    windows: list[np.ndarray] = []
    labels: list[int] = []
    participant_ids: list[str] = []
    trial_ids: list[str] = []

    grouped = frame.groupby(["participant_id", "trial_id"], dropna=False)
    for (participant_id, trial_id), group in grouped:
        group = group.sort_values("timestamp").reset_index(drop=True)
        if len(group) < window_size:
            continue
        values = group[sequence_columns].fillna(0.0).to_numpy(dtype=np.float32)
        label = int(group["label"].mode(dropna=True).iloc[0]) if group["label"].notna().any() else 0
        for start in range(0, len(group) - window_size + 1, stride):
            stop = start + window_size
            windows.append(values[start:stop])
            labels.append(label)
            participant_ids.append(str(participant_id))
            trial_ids.append(str(trial_id))

    if not windows:
        return WindowedDataset(
            X=np.zeros((0, window_size, len(sequence_columns)), dtype=np.float32),
            y=np.zeros((0,), dtype=np.int64),
            participant_ids=np.array([], dtype=object),
            trial_ids=np.array([], dtype=object),
        )

    return WindowedDataset(
        X=np.stack(windows).astype(np.float32),
        y=np.asarray(labels, dtype=np.int64),
        participant_ids=np.asarray(participant_ids, dtype=object),
        trial_ids=np.asarray(trial_ids, dtype=object),
    )
