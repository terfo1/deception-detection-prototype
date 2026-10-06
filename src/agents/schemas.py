from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import TypeAlias


class QualityStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    WARNING = "WARNING"
    REJECTED = "REJECTED"


class VerificationStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    INCONCLUSIVE = "INCONCLUSIVE"
    REJECTED = "REJECTED"


class WorkflowStatus(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"


class Step(str, Enum):
    DATA_ACQUISITION = "DATA_ACQUISITION"
    QUALITY_CHECK = "QUALITY_CHECK"
    FEATURE_EXTRACTION = "FEATURE_EXTRACTION"
    PREDICTION = "PREDICTION"
    VERIFICATION = "VERIFICATION"
    REPORT = "REPORT"


def identifier(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Identifiers must be nonempty strings.")


def ratio(value: float) -> None:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Ratios must be finite and in [0, 1].")


@dataclass(frozen=True)
class SessionRequest:
    session_id: str
    participant_id: str
    trial_id: str
    timestamp_unit: str
    question_id: str | None = None
    statement_id: str | None = None

    def __post_init__(self) -> None:
        for value in (self.session_id, self.participant_id, self.trial_id):
            identifier(value)
        for value in (self.question_id, self.statement_id):
            if value is not None:
                identifier(value)
        if self.timestamp_unit not in {"s", "ms", "us"}:
            raise ValueError("timestamp_unit must be explicitly s, ms or us.")


@dataclass(frozen=True)
class EyeSample:
    timestamp: float | None
    gaze_x: float | None
    gaze_y: float | None
    pupil: float | None = None
    blink: float | None = None
    validity: float | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        for value in asdict(self).values():
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError("Samples use finite numbers or explicit None for invalid/missing values.")


@dataclass(frozen=True)
class EyeTrackingWindow:
    request: SessionRequest
    samples: tuple[EyeSample, ...]
    available_channels: tuple[str, ...]
    stimulus_id: str | None = None
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request, SessionRequest) or not isinstance(self.samples, tuple):
            raise ValueError("Window requires a typed request and a tuple of samples.")
        if any(not isinstance(sample, EyeSample) for sample in self.samples):
            raise ValueError("Window samples must be typed EyeSample messages.")
        allowed = {"timestamp", "gaze_x", "gaze_y", "pupil", "blink", "validity", "confidence"}
        if not set(self.available_channels) <= allowed:
            raise ValueError("Unknown measured channels.")
        if self.source_sha256 is not None:
            if len(self.source_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.source_sha256):
                raise ValueError("Source SHA256 must be lowercase hexadecimal.")

    @property
    def timestamp_range(self) -> tuple[float | None, float | None]:
        times = [sample.timestamp for sample in self.samples if sample.timestamp is not None]
        return (min(times), max(times)) if times else (None, None)


@dataclass(frozen=True)
class WorkflowConfig:
    max_workflow_steps: int = 16
    max_retries: int = 2
    deadline_seconds: float = 60.0
    min_samples: int = 16
    max_missing_ratio: float = 0.2
    validity_min: float | None = 0.5
    tracking_confidence_min: float | None = None
    sampling_jitter_warning: float = 0.2
    expected_sampling_hz: float | None = None
    gaze_bounds: tuple[float, float, float, float] | None = None
    pupil_bounds: tuple[float, float] | None = None
    min_model_confidence: float | None = None

    def __post_init__(self) -> None:
        for name in ("max_workflow_steps", "min_samples"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if type(self.max_retries) is not int or self.max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer.")
        if not math.isfinite(self.deadline_seconds) or self.deadline_seconds <= 0:
            raise ValueError("deadline_seconds must be positive and finite.")
        for value in (self.max_missing_ratio, self.validity_min, self.tracking_confidence_min,
                      self.min_model_confidence):
            if value is not None:
                ratio(value)
        if not math.isfinite(self.sampling_jitter_warning) or self.sampling_jitter_warning < 0:
            raise ValueError("sampling_jitter_warning must be nonnegative and finite.")
        if self.expected_sampling_hz is not None:
            if not math.isfinite(self.expected_sampling_hz) or self.expected_sampling_hz <= 0:
                raise ValueError("expected_sampling_hz must be positive and finite.")
        for bounds, size in ((self.gaze_bounds, 4), (self.pupil_bounds, 2)):
            if bounds is not None:
                if len(bounds) != size or not all(math.isfinite(v) for v in bounds):
                    raise ValueError("Bounds must have the expected number of finite values.")
                if any(bounds[i] >= bounds[i + 1] for i in range(0, size, 2)):
                    raise ValueError("Each lower bound must be less than its upper bound.")


@dataclass(frozen=True)
class QualityReport:
    status: QualityStatus
    quality_score: float
    missing_ratio: float
    valid_samples: int
    total_samples: int
    reasons: tuple[str, ...]
    unavailable_checks: tuple[str, ...]
    sampling_hz: float | None
    sampling_jitter: float | None
    accumulation_time_ms: float

    def __post_init__(self) -> None:
        if not isinstance(self.status, QualityStatus):
            raise ValueError("Quality status must use QualityStatus.")
        ratio(self.quality_score)
        ratio(self.missing_ratio)
        if not 0 <= self.valid_samples <= self.total_samples:
            raise ValueError("Invalid sample counts.")
        for value in (self.sampling_hz, self.sampling_jitter, self.accumulation_time_ms):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("Quality timing values must be finite and nonnegative.")


@dataclass(frozen=True)
class FeatureVector:
    kind: str
    names: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]
    sample_count: int

    def __post_init__(self) -> None:
        if self.kind not in {"aggregated", "sequence"} or not self.names or not self.values:
            raise ValueError("Features require a supported kind, names and nonempty values.")
        if len(set(self.names)) != len(self.names) or "label" in self.names:
            raise ValueError("Feature names must be unique and exclude labels.")
        if self.kind == "aggregated" and len(self.values) != 1:
            raise ValueError("Aggregated features have exactly one row.")
        for row in self.values:
            if len(row) != len(self.names) or not all(math.isfinite(v) for v in row):
                raise ValueError("Feature shape mismatch or non-finite features.")
        if self.sample_count < 1:
            raise ValueError("Features require at least one sample.")


@dataclass(frozen=True)
class PredictionResult:
    predicted_class: int
    model_version: str
    target: str
    class_meanings: tuple[tuple[int, str], ...]
    inference_time_ms: float
    probabilities: tuple[tuple[int, float], ...] | None = None
    confidence: float | None = None
    probability_kind: str | None = None

    def __post_init__(self) -> None:
        identifier(self.model_version)
        if self.target not in {"truth_lie", "cit", "synthetic_debug"}:
            raise ValueError("Explicit model target required.")
        classes = [key for key, meaning in self.class_meanings if isinstance(meaning, str) and meaning.strip()]
        if len(classes) != len(self.class_meanings) or len(set(classes)) != len(classes):
            raise ValueError("Class meanings must be nonempty and unique by class.")
        if self.predicted_class not in classes:
            raise ValueError("Predicted class is absent from the model contract.")
        if not math.isfinite(self.inference_time_ms) or self.inference_time_ms < 0:
            raise ValueError("Invalid inference time.")
        if self.confidence is not None:
            ratio(self.confidence)
        if self.probabilities is not None:
            if [key for key, _ in self.probabilities] != classes:
                raise ValueError("Probability class order must match model classes.")
            for _, probability in self.probabilities:
                ratio(probability)
            if not math.isclose(sum(p for _, p in self.probabilities), 1.0, abs_tol=1e-5):
                raise ValueError("Probabilities must sum to one.")
            selected = dict(self.probabilities)[self.predicted_class]
            if self.confidence is None or not math.isclose(self.confidence, selected, abs_tol=1e-5):
                raise ValueError("Confidence must be the returned probability of the predicted class.")
            if self.probability_kind not in {"predict_proba_uncalibrated", "softmax_uncalibrated"}:
                raise ValueError("Probability provenance required.")
        elif self.confidence is not None or self.probability_kind is not None:
            raise ValueError("Cannot fabricate confidence without probability output.")


@dataclass(frozen=True)
class VerificationResult:
    status: VerificationStatus
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.status, VerificationStatus):
            raise ValueError("Verification status must use VerificationStatus.")


@dataclass(frozen=True)
class WorkflowError:
    code: str
    step: Step
    recoverable: bool = False


@dataclass(frozen=True)
class SessionReport:
    request: SessionRequest
    timestamp_range: tuple[float | None, float | None] | None
    available_channels: tuple[str, ...]
    status: WorkflowStatus
    quality: QualityReport | None
    features: FeatureVector | None
    prediction: PredictionResult | None
    verification: VerificationResult | None
    errors: tuple[WorkflowError, ...]
    warnings: tuple[str, ...]
    retry_count: int
    step_count: int


@dataclass(frozen=True)
class AgentTask:
    session_id: str
    step: Step
    retry_attempt: int

    def __post_init__(self) -> None:
        identifier(self.session_id)
        if not isinstance(self.step, Step) or self.retry_attempt < 0:
            raise ValueError("Task requires a typed step and nonnegative retry attempt.")


Payload: TypeAlias = EyeTrackingWindow | QualityReport | FeatureVector | PredictionResult | VerificationResult | SessionReport


@dataclass(frozen=True)
class AgentResult:
    task: AgentTask
    payload: Payload

    def __post_init__(self) -> None:
        expected = dict(zip(Step, (EyeTrackingWindow, QualityReport, FeatureVector, PredictionResult,
                                  VerificationResult, SessionReport)))
        if not isinstance(self.payload, expected[self.task.step]):
            raise ValueError("Result payload does not match the task step.")


@dataclass
class SessionState:
    request: SessionRequest
    status: WorkflowStatus = WorkflowStatus.CREATED
    current_step: Step = Step.DATA_ACQUISITION
    raw_data: EyeTrackingWindow | None = None
    quality_report: QualityReport | None = None
    features: FeatureVector | None = None
    prediction: PredictionResult | None = None
    verification: VerificationResult | None = None
    report: SessionReport | None = None
    retry_count: int = 0
    step_count: int = 0
    errors: list[WorkflowError] = field(default_factory=list)

    def reset_acquisition(self) -> None:
        self.raw_data = self.quality_report = self.features = self.prediction = self.verification = None
