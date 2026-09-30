from __future__ import annotations

from dataclasses import dataclass, field


CANONICAL_COLUMNS = [
    "participant_id",
    "trial_id",
    "timestamp",
    "gaze_x",
    "gaze_y",
    "pupil",
    "blink",
    "validity",
    "confidence",
    "label",
    "stimulus_id",
]


DEFAULT_ALIASES: dict[str, list[str]] = {
    "participant_id": ["participant_id", "participant", "subject", "subject_id", "recording_id"],
    "trial_id": ["trial_id", "trial", "question", "question_no", "segment_id"],
    "timestamp": ["timestamp", "time", "time_ms", "t", "recording_timestamp"],
    "gaze_x": ["gaze_x", "x", "gaze position x", "gaze_point_x", "fpogx", "screen_x"],
    "gaze_y": ["gaze_y", "y", "gaze position y", "gaze_point_y", "fpogy", "screen_y"],
    "pupil": ["pupil", "pupil_size", "pupil_diameter", "lpd", "rpd", "mean_pupil"],
    "blink": ["blink", "blink_flag", "is_blink", "blink_count"],
    "validity": ["validity", "valid", "quality", "fpogv", "bpogv"],
    "confidence": ["confidence", "conf", "confidence_score"],
    "label": ["label", "class", "group", "condition", "guilty"],
    "stimulus_id": ["stimulus_id", "stimulus", "item", "target", "image_name"],
}


@dataclass(slots=True)
class DatasetSpec:
    """Configuration for a dataset adapter."""

    name: str
    file_pattern: str
    reader: str = "excel"
    aliases: dict[str, list[str]] = field(default_factory=lambda: DEFAULT_ALIASES.copy())
