from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.data.schema import CANONICAL_COLUMNS, DatasetSpec, DEFAULT_ALIASES
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


class TabularEyeTrackingAdapter:
    """Base adapter for heterogeneous tabular eye-tracking exports."""

    def __init__(self, spec: DatasetSpec, allow_missing_schema: bool = True) -> None:
        self.spec = spec
        self.allow_missing_schema = allow_missing_schema

    def discover_files(self, raw_dir: Path, recursive: bool = False) -> list[Path]:
        pattern = f"**/{self.spec.file_pattern}" if recursive else self.spec.file_pattern
        return sorted(raw_dir.glob(pattern))

    def load_file(self, path: Path) -> pd.DataFrame:
        if self.spec.reader == "excel":
            frame = pd.read_excel(path)
        elif self.spec.reader == "csv":
            frame = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported reader: {self.spec.reader}")
        return self.normalize_schema(frame, source_path=path)

    def normalize_schema(self, frame: pd.DataFrame, source_path: Path) -> pd.DataFrame:
        renamed = {}
        normalized_lookup = {_normalize_name(column): column for column in frame.columns}
        for canonical, aliases in self.spec.aliases.items():
            for alias in aliases:
                match = normalized_lookup.get(_normalize_name(alias))
                if match is not None:
                    renamed[match] = canonical
                    break

        frame = frame.rename(columns=renamed).copy()
        for column in CANONICAL_COLUMNS:
            if column not in frame.columns:
                frame[column] = pd.NA

        frame = self.postprocess_schema(frame, source_path)

        missing = [column for column in CANONICAL_COLUMNS if frame[column].isna().all()]
        if missing:
            message = f"{source_path.name}: unresolved or empty canonical columns {missing}"
            if self.allow_missing_schema:
                LOGGER.warning(message)
            else:
                raise ValueError(message)

        frame["source_file"] = source_path.name
        if frame["participant_id"].isna().all():
            frame["participant_id"] = source_path.stem
        if frame["trial_id"].isna().all():
            frame["trial_id"] = 0
        if frame["label"].isna().all():
            frame["label"] = self.infer_default_label(source_path)
        return frame

    def postprocess_schema(self, frame: pd.DataFrame, source_path: Path) -> pd.DataFrame:
        """Hook for dataset-specific column derivation after alias renaming."""
        return frame

    def infer_default_label(self, source_path: Path) -> int:
        LOGGER.warning("%s: label missing; defaulting to 0 placeholder.", source_path.name)
        return 0


class CITAdapter(TabularEyeTrackingAdapter):
    """Adapter for the Zenodo CIT spreadsheet recordings."""

    def __init__(self, allow_missing_schema: bool = True) -> None:
        aliases = DEFAULT_ALIASES.copy()
        aliases["participant_id"] = DEFAULT_ALIASES["participant_id"] + ["participant name", "recording name"]
        aliases["label"] = DEFAULT_ALIASES["label"] + ["group_assignment", "question_type", "group"]
        aliases["trial_id"] = DEFAULT_ALIASES["trial_id"] + ["trial_number"]
        aliases["timestamp"] = DEFAULT_ALIASES["timestamp"] + ["eyetracker timestamp", "recording timestamp"]
        aliases["gaze_x"] = DEFAULT_ALIASES["gaze_x"] + ["gaze point x", "gaze point x (mcsnorm)"]
        aliases["gaze_y"] = DEFAULT_ALIASES["gaze_y"] + ["gaze point y", "gaze point y (mcsnorm)"]
        aliases["pupil"] = DEFAULT_ALIASES["pupil"] + ["pupil diameter filtered", "pupil diameter left", "pupil diameter right"]
        aliases["validity"] = DEFAULT_ALIASES["validity"] + ["validity left", "validity right"]
        aliases["stimulus_id"] = DEFAULT_ALIASES["stimulus_id"] + ["stimulus_name", "presented stimulus name", "presented media name"]
        super().__init__(
            DatasetSpec(name="cit", file_pattern="*.xlsx", reader="excel", aliases=aliases),
            allow_missing_schema=allow_missing_schema,
        )

    def postprocess_schema(self, frame: pd.DataFrame, source_path: Path) -> pd.DataFrame:
        label_map = {
            "guilty": 1,
            "concealed": 1,
            "target": 1,
            "innocent": 0,
            "control": 0,
            "non_guilty": 0,
            "nontarget": 0,
        }

        if "Participant name" in frame.columns:
            frame["participant_id"] = frame["participant_id"].fillna(frame["Participant name"])
        if "Recording name" in frame.columns:
            frame["participant_id"] = frame["participant_id"].fillna(frame["Recording name"])

        if frame["label"].notna().any():
            mapped_labels = frame["label"].astype(str).str.strip().str.lower().map(label_map)
            numeric_labels = pd.to_numeric(frame["label"], errors="coerce")
            frame["label"] = mapped_labels.where(mapped_labels.notna(), numeric_labels)

        if "Eyetracker timestamp" in frame.columns:
            frame["timestamp"] = frame["timestamp"].where(frame["timestamp"].notna(), frame["Eyetracker timestamp"])
        if "Recording timestamp" in frame.columns:
            frame["timestamp"] = frame["timestamp"].where(frame["timestamp"].notna(), frame["Recording timestamp"])

        if "Pupil diameter filtered" in frame.columns:
            frame["pupil"] = frame["pupil"].where(frame["pupil"].notna(), frame["Pupil diameter filtered"])
        if "Pupil diameter left" in frame.columns or "Pupil diameter right" in frame.columns:
            left = frame["Pupil diameter left"] if "Pupil diameter left" in frame.columns else pd.Series(pd.NA, index=frame.index)
            right = frame["Pupil diameter right"] if "Pupil diameter right" in frame.columns else pd.Series(pd.NA, index=frame.index)
            combined = pd.concat([left, right], axis=1).apply(pd.to_numeric, errors="coerce").mean(axis=1)
            frame["pupil"] = frame["pupil"].where(frame["pupil"].notna(), combined)

        validity_left = frame["Validity left"] if "Validity left" in frame.columns else pd.Series(pd.NA, index=frame.index)
        validity_right = frame["Validity right"] if "Validity right" in frame.columns else pd.Series(pd.NA, index=frame.index)
        validity_map = {"valid": 1.0, "invalid": 0.0}
        if validity_left.notna().any() or validity_right.notna().any():
            left_num = validity_left.astype(str).str.strip().str.lower().map(validity_map)
            right_num = validity_right.astype(str).str.strip().str.lower().map(validity_map)
            combined_validity = pd.concat([left_num, right_num], axis=1).mean(axis=1)
            frame["validity"] = frame["validity"].where(frame["validity"].notna(), combined_validity)
        if frame["validity"].notna().any():
            mapped_validity = frame["validity"].astype(str).str.strip().str.lower().map(validity_map)
            numeric_validity = pd.to_numeric(frame["validity"], errors="coerce")
            frame["validity"] = mapped_validity.where(mapped_validity.notna(), numeric_validity)

        if "Presented Stimulus name" in frame.columns:
            frame["stimulus_id"] = frame["stimulus_id"].where(frame["stimulus_id"].notna(), frame["Presented Stimulus name"])
        if "Presented Media name" in frame.columns:
            frame["stimulus_id"] = frame["stimulus_id"].where(frame["stimulus_id"].notna(), frame["Presented Media name"])

        frame["participant_id"] = frame["participant_id"].fillna(source_path.stem)

        stimulus = frame["stimulus_id"].astype("string").fillna("<NA>")
        meaningful = stimulus.ne("<NA>") & stimulus.ne("")
        trial_switch = meaningful & stimulus.ne(stimulus.shift(fill_value=stimulus.iloc[0]))
        frame["trial_id"] = frame["trial_id"].where(frame["trial_id"].notna(), trial_switch.cumsum())

        return frame

    def infer_default_label(self, source_path: Path) -> int:
        name = source_path.stem.lower()
        if "guilty" in name or "target" in name:
            return 1
        return 0


class BagOfLiesAdapter(TabularEyeTrackingAdapter):
    """Placeholder adapter for future Bag-of-Lies tabular eye-tracking exports."""

    def __init__(self, allow_missing_schema: bool = True) -> None:
        super().__init__(
            DatasetSpec(name="bag_of_lies", file_pattern="*.csv", reader="csv"),
            allow_missing_schema=allow_missing_schema,
        )


def iter_loaded_frames(adapter: TabularEyeTrackingAdapter, files: Iterable[Path]) -> Iterable[pd.DataFrame]:
    """Load files lazily through a given adapter."""
    for path in files:
        yield adapter.load_file(path)
