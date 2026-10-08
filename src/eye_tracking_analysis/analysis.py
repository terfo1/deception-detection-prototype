from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .validation import IDS, validate_data

EVENT_COLUMNS = IDS + ["event_type", "start_s", "end_s", "duration_s", "amplitude", "peak_velocity", "mean_velocity"]


@dataclass(frozen=True)
class AnalysisConfig:
    coordinate_unit: str
    velocity_threshold: float
    timestamp_unit: str | None = None
    min_fixation_s: float = 0.1
    max_gap_s: float = 0.1
    interpolate_max_s: float = 0.0
    smoothing_ms: float = 0.0
    screen_size: tuple[float, float] | None = None

    def __post_init__(self):
        if self.coordinate_unit not in {"pixels", "degrees", "normalized"}:
            raise ValueError("coordinate_unit must be pixels, degrees, or normalized")
        if self.timestamp_unit not in {None, "s", "ms"}:
            raise ValueError("timestamp_unit must be s or ms")
        for name in ["velocity_threshold", "min_fixation_s", "max_gap_s"]:
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ["interpolate_max_s", "smoothing_ms"]:
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.screen_size is not None:
            if self.coordinate_unit != "pixels" or len(self.screen_size) != 2:
                raise ValueError("screen_size requires pixel coordinates and width/height")
            if any(not np.isfinite(v) or v <= 0 for v in self.screen_size):
                raise ValueError("screen_size must be finite and positive")

    @property
    def output_unit(self) -> str:
        return "normalized" if self.screen_size is not None else self.coordinate_unit


@dataclass
class AnalysisResult:
    samples: pd.DataFrame
    features: pd.DataFrame
    events: pd.DataFrame
    metadata: dict[str, Any]


def _runs(mask: np.ndarray):
    changes = np.diff(np.r_[False, mask, False].astype(int))
    return zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))


def preprocess_group(group: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    """Bounded offline interpolation and trailing time-window smoothing within a trial."""
    group = group.copy().reset_index(drop=True)
    xy = group[["gaze_x", "gaze_y"]].to_numpy(copy=True)
    blink = group["blink"].to_numpy() == 1 if "blink" in group else np.zeros(len(group), dtype=bool)
    missing = ~np.isfinite(xy).all(axis=1)
    group["gaze_missing_input"] = missing
    xy[missing | blink] = np.nan
    filled = np.zeros(len(group), dtype=bool)
    if "time_s" in group:
        t = group["time_s"].to_numpy()
        for start, end in _runs(missing & ~blink):
            if start == 0 or end == len(group) or not np.isfinite(xy[[start - 1, end]]).all():
                continue
            span = t[end] - t[start - 1]
            if span > config.interpolate_max_s or np.any(np.diff(t[start - 1:end + 1]) > config.max_gap_s):
                continue
            weight = ((t[start:end] - t[start - 1]) / span)[:, None]
            xy[start:end] = xy[start - 1] + weight * (xy[end] - xy[start - 1])
            filled[start:end] = True
        if config.smoothing_ms:
            # Reset on unfilled missingness, observed blinks and timestamp gaps.
            valid = np.isfinite(xy).all(axis=1)
            breaks = np.r_[True, (~valid[1:]) | (~valid[:-1]) | (np.diff(t) > config.max_gap_s)]
            segment_start = 0
            original = xy.copy()
            for i in range(len(group)):
                if breaks[i]:
                    segment_start = i
                if valid[i]:
                    left = max(segment_start, np.searchsorted(t, t[i] - config.smoothing_ms / 1000, side="right"))
                    xy[i] = original[left:i + 1].mean(axis=0)
    if config.screen_size is not None:
        xy /= np.asarray(config.screen_size)
    group[["gaze_x", "gaze_y"]] = xy
    group["gaze_interpolated"] = filled
    group["gaze_usable"] = np.isfinite(xy).all(axis=1)
    return group


def extract_group(group: pd.DataFrame, config: AnalysisConfig) -> tuple[dict, list[dict]]:
    ids = {name: group[name].iloc[0] for name in IDS}
    xy = group[["gaze_x", "gaze_y"]].to_numpy()
    valid = np.isfinite(xy).all(axis=1)
    clean = xy[valid]
    features = {
        **ids, "n_samples": len(group), "usable_samples": int(valid.sum()),
        "missing_input_fraction": float(group["gaze_missing_input"].mean()),
        "interpolated_samples": int(group["gaze_interpolated"].sum()),
        "gaze_dispersion": float(np.sqrt(np.var(clean, axis=0).sum())) if len(clean) else None,
        "duration_s": None, "observed_duration_s": None, "sampling_rate_hz_estimate": None,
        "timestamp_gap_count": None, "fixation_count": None, "mean_fixation_duration_s": None,
        "saccade_count": None, "mean_saccade_amplitude": None, "mean_saccade_peak_velocity": None,
        "blink_onset_count": None, "blink_rate_per_min": None,
    }
    events = []
    if "time_s" not in group or len(group) < 2:
        return features, events
    t = group["time_s"].to_numpy()
    dt = np.diff(t)
    observed = dt <= config.max_gap_s
    duration = float(dt[observed].sum())
    features.update(duration_s=float(t[-1] - t[0]), observed_duration_s=duration,
                    sampling_rate_hz_estimate=float(1 / np.median(dt)),
                    timestamp_gap_count=int((~observed).sum()))
    if "blink" in group:
        b = group["blink"].to_numpy()
        onsets = int(((b[:-1] == 0) & (b[1:] == 1) & observed).sum())
        features.update(blink_onset_count=onsets, blink_rate_per_min=60 * onsets / duration if duration else None)
    usable = valid[:-1] & valid[1:] & observed
    velocity = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
    for kind, mask in [("fixation", usable & (velocity <= config.velocity_threshold)),
                       ("saccade", usable & (velocity > config.velocity_threshold))]:
        for start, end in _runs(mask):
            event_duration = float(t[end] - t[start])
            if kind == "fixation" and event_duration < config.min_fixation_s:
                continue
            events.append({**ids, "event_type": kind, "start_s": float(t[start]), "end_s": float(t[end]),
                           "duration_s": event_duration,
                           "amplitude": float(np.linalg.norm(xy[end] - xy[start])),
                           "peak_velocity": float(velocity[start:end].max()),
                           "mean_velocity": float(np.sum(velocity[start:end] * dt[start:end]) / event_duration)})
    fixations = [e for e in events if e["event_type"] == "fixation"]
    saccades = [e for e in events if e["event_type"] == "saccade"]
    features.update(fixation_count=len(fixations), saccade_count=len(saccades),
                    mean_fixation_duration_s=float(np.mean([e["duration_s"] for e in fixations])) if fixations else None,
                    mean_saccade_amplitude=float(np.mean([e["amplitude"] for e in saccades])) if saccades else None,
                    mean_saccade_peak_velocity=float(np.mean([e["peak_velocity"] for e in saccades])) if saccades else None)
    return features, events


def analyze(frame: pd.DataFrame, config: AnalysisConfig) -> AnalysisResult:
    validated = validate_data(frame)
    timed = "timestamp" in validated
    if timed and config.timestamp_unit is None:
        raise ValueError("Explicit timestamp_unit required when timestamp is present")
    if not timed and (config.interpolate_max_s or config.smoothing_ms):
        raise ValueError("Temporal preprocessing requires timestamps")
    if timed:
        validated["time_s"] = validated["timestamp"] * (0.001 if config.timestamp_unit == "ms" else 1.0)
        if not np.isfinite(validated["time_s"]).all():
            raise ValueError("Non-finite time in seconds")
    samples, features, events = [], [], []
    for _, group in validated.groupby(IDS, sort=False):
        processed = preprocess_group(group, config)
        feature, trial_events = extract_group(processed, config)
        samples.append(processed)
        features.append(feature)
        events.extend(trial_events)
    metadata = {"schema_version": 1, "mode": "offline", "config": asdict(config),
                "coordinate_unit": config.output_unit, "velocity_unit": f"{config.output_unit}/s",
                "source": dict(frame.attrs), "labels_used": False,
                "limitations": ["Engineering velocity threshold; no validated physiological event detector",
                                "Interpolation uses future samples; not a streaming pipeline",
                                "Missing gaze is not evidence of blinking",
                                "Blink onset at recording start or after a gap is left-censored",
                                "No truth/lie prediction or scientific accuracy claim"]}
    return AnalysisResult(pd.concat(samples, ignore_index=True), pd.DataFrame(features),
                          pd.DataFrame(events, columns=EVENT_COLUMNS), metadata)
