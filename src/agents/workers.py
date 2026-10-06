from __future__ import annotations

from abc import ABC, abstractmethod

from src.agents.schemas import AgentResult, AgentTask, Payload, SessionState, Step
from src.agents.services import (
    EyeTrackingSource, FeatureExtractionService, ModelInferenceService, ReportService,
    SignalQualityService, VerificationService,
)
from src.agents.telemetry import StageFailure, ToolContext


class BaseAgent(ABC):
    name: str
    role: str
    step: Step
    tools: tuple[str, ...]

    def execute(self, task: AgentTask, state: SessionState, context: ToolContext) -> AgentResult:
        if task.step != self.step or task.session_id != state.request.session_id:
            raise StageFailure("AGENT_TASK_MISMATCH")
        with context.journal.execution(self.name, task):
            return AgentResult(task, self.perform(state, context))

    @abstractmethod
    def perform(self, state: SessionState, context: ToolContext) -> Payload: ...


class DataAcquisitionAgent(BaseAgent):
    name, role, step = "DataAcquisitionAgent", "Acquire one bounded trial", Step.DATA_ACQUISITION
    tools = ("source.acquire",)

    def __init__(self, source: EyeTrackingSource) -> None:
        self.source = source

    def perform(self, state: SessionState, context: ToolContext) -> Payload:
        window = context.call(self.tools[0], self.source.acquire, state.request)
        if window.request != state.request:
            raise StageFailure("SOURCE_REQUEST_MISMATCH")
        return window


class SignalQualityAgent(BaseAgent):
    name, role, step = "SignalQualityAgent", "Gate measured signal quality", Step.QUALITY_CHECK
    tools = ("quality.assess",)

    def __init__(self, service: SignalQualityService) -> None:
        self.service = service

    def perform(self, state: SessionState, context: ToolContext) -> Payload:
        if state.raw_data is None:
            raise StageFailure("QUALITY_INPUT_MISSING")
        return context.call(self.tools[0], self.service.assess, state.raw_data)


class FeatureExtractionAgent(BaseAgent):
    name, role, step = "FeatureExtractionAgent", "Invoke existing processing and features", Step.FEATURE_EXTRACTION
    tools = ("features.preprocess", "features.extract")

    def __init__(self, service: FeatureExtractionService) -> None:
        self.service = service

    def perform(self, state: SessionState, context: ToolContext) -> Payload:
        if state.raw_data is None:
            raise StageFailure("FEATURE_INPUT_MISSING")
        frame = context.call(self.tools[0], self.service.preprocess, state.raw_data)
        return context.call(self.tools[1], self.service.extract, frame)


class PredictionAgent(BaseAgent):
    name, role, step = "PredictionAgent", "Invoke saved model inference", Step.PREDICTION
    tools = ("model.predict",)

    def __init__(self, service: ModelInferenceService) -> None:
        self.service = service

    def perform(self, state: SessionState, context: ToolContext) -> Payload:
        if state.features is None:
            raise StageFailure("PREDICTION_INPUT_MISSING")
        return context.call(self.tools[0], self.service.predict, state.features)


class VerificationAgent(BaseAgent):
    name, role, step = "VerificationAgent", "Apply explicit decision policy", Step.VERIFICATION
    tools = ("verification.verify",)

    def __init__(self, service: VerificationService) -> None:
        self.service = service

    def perform(self, state: SessionState, context: ToolContext) -> Payload:
        return context.call(self.tools[0], self.service.verify, state)


class ReportAgent(BaseAgent):
    name, role, step = "ReportAgent", "Assemble results without inference", Step.REPORT
    tools = ("report.build",)

    def __init__(self, service: ReportService) -> None:
        self.service = service

    def perform(self, state: SessionState, context: ToolContext) -> Payload:
        return context.call(self.tools[0], self.service.build, state)
