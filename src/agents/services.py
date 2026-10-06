from __future__ import annotations

from dataclasses import asdict, dataclass, field
import errno
import hashlib
import json
import math
from pathlib import Path
import pickle
import time
from typing import Protocol

import numpy as np
import pandas as pd

from src.agents.schemas import (
    EyeSample, EyeTrackingWindow, FeatureVector, PredictionResult, QualityReport, QualityStatus,
    SessionReport, SessionRequest, SessionState, VerificationResult, VerificationStatus, WorkflowConfig,
)
from src.agents.telemetry import StageFailure
from src.data.adapters import CITAdapter, TabularEyeTrackingAdapter
from src.data.schema import DatasetSpec
from src.features.aggregated import create_aggregated_features
from src.features.preprocessing import preprocess_eye_tracking

CHANNELS = ("timestamp", "gaze_x", "gaze_y", "pupil", "blink", "validity", "confidence")


class EyeTrackingSource(Protocol):
    can_refresh: bool

    def acquire(self, request: SessionRequest) -> EyeTrackingWindow: ...


class FileEyeTrackingSource:
    """Single explicitly selected trial/session; no synthetic or label fallback."""

    can_refresh = False

    def __init__(self, path: Path, adapter: TabularEyeTrackingAdapter | None = None) -> None:
        self.path = Path(path)
        self.source_sha256: str | None = None
        self.adapter = adapter or TabularEyeTrackingAdapter(
            DatasetSpec("inference", "*.csv", "csv"), allow_missing_schema=True
        )

    def acquire(self, request: SessionRequest) -> EyeTrackingWindow:
        try:
            before = hashlib.sha256(self.path.read_bytes()).hexdigest()
            frame = self.adapter.load_file(self.path, for_inference=True)
            after = hashlib.sha256(self.path.read_bytes()).hexdigest()
            if before != after:
                raise StageFailure("SOURCE_CHANGED_DURING_ACQUISITION")
            self.source_sha256 = before
        except OSError as exc:
            recoverable = exc.errno in {errno.EAGAIN, errno.EINTR, errno.ETIMEDOUT}
            raise StageFailure("SOURCE_IO_ERROR", recoverable=recoverable) from exc
        # Explicit IDs are compared as text, including numeric CSV trial IDs.
        mask = (frame["participant_id"].astype("string") == request.participant_id)
        mask &= frame["trial_id"].astype("string") == request.trial_id
        for column, value in (("session_id", request.session_id), ("question_id", request.question_id),
                              ("statement_id", request.statement_id)):
            if column in frame:
                if value is not None:
                    mask &= frame[column].astype("string") == value
                else:
                    raise StageFailure("SOURCE_BOUNDARY_SELECTION_REQUIRED")
            elif value is not None and column != "session_id":
                raise StageFailure("SOURCE_BOUNDARY_ID_MISSING")
        positions = np.flatnonzero(mask.fillna(False).to_numpy(dtype=bool))
        if not len(positions):
            raise StageFailure("SOURCE_TRIAL_NOT_FOUND")
        if len(positions) > 1 and np.any(np.diff(positions) != 1):
            raise StageFailure("SOURCE_DISJOINT_TRIAL")
        selected = frame.iloc[positions].copy()
        for column in ("session_id", "question_id", "statement_id", "stimulus_id"):
            if column in selected:
                if selected[column].nunique(dropna=False) > 1:
                    raise StageFailure("SOURCE_MIXED_BOUNDARIES")
                if column != "stimulus_id" and selected[column].isna().any():
                    raise StageFailure("SOURCE_BOUNDARY_ID_MISSING")
        available = tuple(column for column in CHANNELS if selected[column].notna().any())
        numeric = selected[list(CHANNELS)].apply(pd.to_numeric, errors="coerce")
        samples = tuple(EyeSample(**{column: float(value) if pd.notna(value) and math.isfinite(value) else None
                                    for column, value in zip(CHANNELS, row)})
                        for row in numeric.itertuples(index=False, name=None))
        stimulus = selected["stimulus_id"].dropna()
        return EyeTrackingWindow(request, samples, available, str(stimulus.iloc[0]) if len(stimulus) else None, before)


@dataclass(frozen=True)
class ModelSpec:
    family: str
    checkpoint: str
    checkpoint_sha256: str
    model_version: str
    target: str
    class_meanings: tuple[tuple[int, str], ...]
    timestamp_unit: str
    preprocessing: dict = field(default_factory=dict)
    sequence_columns: tuple[str, ...] = ()
    window_size: int | None = None
    architecture: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.family not in {"baseline", "sequence"}:
            raise ValueError("Unsupported model family.")
        if len(self.checkpoint_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.checkpoint_sha256):
            raise ValueError("Model contract requires its checkpoint SHA256.")
        SessionRequest("validation", "validation", "validation", self.timestamp_unit)
        if not self.class_meanings or any(type(key) is not int for key, _ in self.class_meanings):
            raise ValueError("Class IDs must be explicit integers.")
        PredictionResult(self.class_meanings[0][0], self.model_version, self.target, self.class_meanings, 0.0)
        if self.family == "sequence":
            if type(self.window_size) is not int or self.window_size < 1 or not self.sequence_columns:
                raise ValueError("Sequence models require window size and ordered channels.")
            if len(set(self.sequence_columns)) != len(self.sequence_columns):
                raise ValueError("Sequence channels must be unique.")
            if any(name not in CHANNELS or name == "timestamp" for name in self.sequence_columns):
                raise ValueError("Sequence channels must be measured signal channels, excluding labels/IDs.")
            if self.architecture.get("input_dim") != len(self.sequence_columns):
                raise ValueError("Architecture input_dim must match channel order.")
            if self.architecture.get("num_classes") != len(self.class_meanings):
                raise ValueError("Architecture num_classes must match class meanings.")

    @classmethod
    def load(cls, path: Path) -> ModelSpec:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["class_meanings"] = tuple((int(key), value) for key, value in data["class_meanings"].items())
        data["sequence_columns"] = tuple(data.get("sequence_columns", ()))
        checkpoint = Path(data["checkpoint"])
        data["checkpoint"] = str(checkpoint if checkpoint.is_absolute() else path.parent / checkpoint)
        return cls(**data)


class SignalQualityService:
    def __init__(self, config: WorkflowConfig, required_channels: tuple[str, ...]) -> None:
        self.config, self.required_channels = config, required_channels

    def assess(self, window: EyeTrackingWindow) -> QualityReport:
        cfg, samples = self.config, window.samples
        total = len(samples)
        reasons, unavailable = [], []
        for channel in ("validity", "confidence", "pupil", "blink"):
            if channel not in window.available_channels:
                unavailable.append(f"{channel}_unavailable")
        if cfg.gaze_bounds is None:
            unavailable.append("coordinate_bounds_unspecified")
        if cfg.pupil_bounds is None:
            unavailable.append("pupil_upper_bound_unspecified")
        if cfg.tracking_confidence_min is not None and "confidence" not in window.available_channels:
            reasons.append("required_tracking_confidence_unavailable")
        valid = 0
        for sample in samples:
            good = all(getattr(sample, name) is not None for name in self.required_channels)
            if "validity" in window.available_channels and cfg.validity_min is not None:
                good &= sample.validity is not None and cfg.validity_min <= sample.validity <= 1
            if "confidence" in window.available_channels:
                good &= sample.confidence is not None and 0 <= sample.confidence <= 1
                if cfg.tracking_confidence_min is not None:
                    good &= sample.confidence is not None and sample.confidence >= cfg.tracking_confidence_min
            if sample.pupil is not None:
                good &= sample.pupil > 0
                if cfg.pupil_bounds is not None:
                    good &= cfg.pupil_bounds[0] <= sample.pupil <= cfg.pupil_bounds[1]
            if cfg.gaze_bounds is not None:
                xmin, xmax, ymin, ymax = cfg.gaze_bounds
                good &= sample.gaze_x is not None and xmin <= sample.gaze_x <= xmax
                good &= sample.gaze_y is not None and ymin <= sample.gaze_y <= ymax
            valid += int(good)
        missing = 1 - valid / total if total else 1.0
        if valid < cfg.min_samples:
            reasons.append("too_few_valid_samples")
        if missing > cfg.max_missing_ratio:
            reasons.append("excess_missing_or_invalid_samples")
        times = np.array([s.timestamp if s.timestamp is not None else np.nan for s in samples], dtype=float)
        hz, jitter, accumulation = None, None, 0.0
        if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            reasons.append("timestamps_missing_or_not_strictly_increasing")
        else:
            scale = {"s": 1, "ms": 0.001, "us": 0.000001}[window.request.timestamp_unit]
            dt = np.diff(times) * scale
            median = float(np.median(dt))
            hz, jitter = 1 / median, float(np.std(dt) / median)
            accumulation = float((times[-1] - times[0]) * scale * 1000)
        rejected = bool(reasons)
        if jitter is not None and jitter > cfg.sampling_jitter_warning:
            reasons.append("irregular_sampling")
        if cfg.expected_sampling_hz is not None and hz is not None:
            if abs(hz / cfg.expected_sampling_hz - 1) > cfg.sampling_jitter_warning:
                reasons.append("sampling_rate_mismatch")
        if missing > 0 and not rejected:
            reasons.append("some_samples_missing_or_invalid")
        status = QualityStatus.REJECTED if rejected else QualityStatus.WARNING if reasons else QualityStatus.ACCEPTED
        return QualityReport(status, 1 - missing, missing, valid, total, tuple(reasons), tuple(unavailable),
                             hz, jitter, accumulation)


class FeatureExtractionService:
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec

    def preprocess(self, window: EyeTrackingWindow) -> pd.DataFrame:
        if window.request.timestamp_unit != self.spec.timestamp_unit:
            raise StageFailure("TIMESTAMP_UNIT_MODEL_MISMATCH")
        frame = pd.DataFrame([asdict(sample) for sample in window.samples])
        frame["participant_id"] = window.request.participant_id
        frame["trial_id"] = window.request.trial_id
        frame["stimulus_id"] = window.stimulus_id
        return preprocess_eye_tracking(frame, self.spec.preprocessing)

    def extract(self, frame: pd.DataFrame) -> FeatureVector:
        if frame.empty:
            raise StageFailure("NO_SAMPLES_AFTER_PREPROCESSING")
        if self.spec.family == "baseline":
            features = create_aggregated_features(frame, include_labels=False)
            names = tuple(column for column in features if column not in {"participant_id", "trial_id", "stimulus_id"})
            values = features[list(names)].fillna(0).to_numpy(dtype=float)
            kind = "aggregated"
        else:
            if len(frame) != self.spec.window_size:
                raise StageFailure("SEQUENCE_WINDOW_SIZE_MISMATCH")
            names, kind = self.spec.sequence_columns, "sequence"
            values = frame[list(names)].fillna(0).to_numpy(dtype=float)
        return FeatureVector(kind, names, tuple(tuple(float(v) for v in row) for row in values), len(frame))


class ModelInferenceService:
    """Load only a trusted checkpoint explicitly supplied by the caller. Never train."""

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec

    def predict(self, features: FeatureVector) -> PredictionResult:
        start = time.perf_counter()
        spec = self.spec
        checkpoint = Path(spec.checkpoint)
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != spec.checkpoint_sha256:
            raise StageFailure("MODEL_CHECKSUM_MISMATCH")
        probabilities, confidence, probability_kind = None, None, None
        classes = tuple(key for key, _ in spec.class_meanings)
        if spec.family == "baseline":
            if features.kind != "aggregated":
                raise StageFailure("MODEL_FEATURE_KIND_MISMATCH")
            with checkpoint.open("rb") as handle:
                bundle = pickle.load(handle)
            columns = tuple(bundle["feature_columns"])
            if not columns or len(set(columns)) != len(columns) or any(name not in features.names for name in columns):
                raise StageFailure("MODEL_FEATURE_SCHEMA_MISMATCH")
            x = np.asarray(features.values)[:, [features.names.index(name) for name in columns]]
            model = bundle["model"]
            actual_classes = tuple(model.classes_.tolist())
            if actual_classes != classes:
                raise StageFailure("MODEL_CLASS_CONTRACT_MISMATCH")
            predicted = np.asarray(model.predict(x))
            if predicted.shape != (1,) or predicted[0] not in classes:
                raise StageFailure("MODEL_INVALID_PREDICTION")
            predicted_class = int(predicted[0])
            if hasattr(model, "predict_proba"):
                raw = np.asarray(model.predict_proba(x))
                if raw.shape != (1, len(classes)):
                    raise StageFailure("MODEL_INVALID_PROBABILITIES")
                probabilities = tuple(zip(classes, (float(v) for v in raw[0])))
                confidence = dict(probabilities)[predicted_class]
                probability_kind = "predict_proba_uncalibrated"
        else:
            import torch
            from src.models.sequence import build_sequence_model

            if features.kind != "sequence" or features.names != spec.sequence_columns:
                raise StageFailure("MODEL_FEATURE_SCHEMA_MISMATCH")
            if len(features.values) != spec.window_size:
                raise StageFailure("SEQUENCE_WINDOW_SIZE_MISMATCH")
            model = build_sequence_model(spec.architecture)
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            model.eval()
            with torch.inference_mode():
                logits = model(torch.tensor(features.values, dtype=torch.float32).unsqueeze(0))
                raw = torch.softmax(logits, dim=1).numpy()
            if raw.shape != (1, len(classes)):
                raise StageFailure("MODEL_INVALID_PROBABILITIES")
            predicted_class = classes[int(raw[0].argmax())]
            probabilities = tuple(zip(classes, (float(v) for v in raw[0])))
            confidence = dict(probabilities)[predicted_class]
            probability_kind = "softmax_uncalibrated"
        return PredictionResult(predicted_class, spec.model_version, spec.target, spec.class_meanings,
                                (time.perf_counter() - start) * 1000, probabilities, confidence, probability_kind)


class VerificationService:
    def __init__(self, config: WorkflowConfig) -> None:
        self.config = config

    def verify(self, state: SessionState) -> VerificationResult:
        quality, prediction = state.quality_report, state.prediction
        if quality is None or prediction is None or quality.status == QualityStatus.REJECTED:
            return VerificationResult(VerificationStatus.REJECTED, ("missing_prediction_or_rejected_signal",))
        reasons = []
        if state.errors:
            reasons.append("recovered_errors_present")
        if quality.status == QualityStatus.WARNING:
            reasons.append("signal_quality_warning")
        threshold = self.config.min_model_confidence
        if prediction.confidence is None:
            reasons.append("model_confidence_unavailable")
        elif threshold is not None and prediction.confidence < threshold:
            reasons.append("below_configured_model_confidence")
        if threshold is None:
            reasons.append("model_confidence_policy_unspecified")
        return VerificationResult(VerificationStatus.INCONCLUSIVE if reasons else VerificationStatus.ACCEPTED,
                                  tuple(reasons) or ("engineering_checks_passed_not_scientific_validation",))


class ReportService:
    def build(self, state: SessionState) -> SessionReport:
        warnings = ["offline_window_analysis_not_physical_live_validation",
                    "model_output_is_not_a_claim_about_person_truthfulness",
                    "quality_and_verification_thresholds_are_engineering_rules"]
        if state.prediction and state.prediction.target == "synthetic_debug":
            warnings.append("synthetic_debug_not_scientific_evidence")
        if state.quality_report:
            warnings.extend(state.quality_report.unavailable_checks)
        return SessionReport(state.request, state.raw_data.timestamp_range if state.raw_data else None,
                             state.raw_data.available_channels if state.raw_data else (),
                             state.status, state.quality_report, state.features, state.prediction,
                             state.verification, tuple(state.errors), tuple(warnings), state.retry_count, state.step_count)


def build_services(source: EyeTrackingSource, spec: ModelSpec, config: WorkflowConfig) -> tuple:
    required = ("timestamp", "gaze_x", "gaze_y", "pupil") if spec.family == "baseline" else (
        "timestamp", *spec.sequence_columns
    )
    return (source, SignalQualityService(config, required), FeatureExtractionService(spec),
            ModelInferenceService(spec), VerificationService(config), ReportService())


def cit_file_source(path: Path) -> FileEyeTrackingSource:
    return FileEyeTrackingSource(path, CITAdapter())
