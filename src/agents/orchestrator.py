from __future__ import annotations

import time
from uuid import uuid4

from src.agents.schemas import (
    AgentTask, QualityStatus, SessionRequest, SessionState, Step, WorkflowConfig, WorkflowError, WorkflowStatus,
)
from src.agents.services import EyeTrackingSource, ModelSpec, build_services
from src.agents.telemetry import ExecutionJournal, StageFailure, ToolContext
from src.agents.workers import (
    DataAcquisitionAgent, FeatureExtractionAgent, PredictionAgent, ReportAgent, SignalQualityAgent, VerificationAgent,
)


class Orchestrator:
    """Bounded state machine. Scientific operations belong exclusively to tools."""

    name = "Orchestrator"
    role = "Coordinate transitions, retries and terminal states"
    tools: tuple[str, ...] = ()

    def __init__(self, source: EyeTrackingSource, spec: ModelSpec, config: WorkflowConfig | None = None) -> None:
        self.config = config or WorkflowConfig()
        self.source = source
        services = build_services(source, spec, self.config)
        agents = [agent(service) for agent, service in zip(
            (DataAcquisitionAgent, SignalQualityAgent, FeatureExtractionAgent, PredictionAgent,
             VerificationAgent, ReportAgent), services
        )]
        self.agents = {agent.step: agent for agent in agents}
        self.state: SessionState | None = None
        self.journal = ExecutionJournal(uuid4().hex)

    def run(self, request: SessionRequest) -> SessionState:
        if self.state is not None:
            raise ValueError("Create a new Orchestrator for each workflow; completed runs cannot be rerun.")
        state = self.state = SessionState(request, status=WorkflowStatus.RUNNING)
        deadline = time.monotonic() + self.config.deadline_seconds
        root_task = AgentTask(request.session_id, Step.DATA_ACQUISITION, 0)
        try:
            with self.journal.execution(self.name, root_task):
                self._run_steps(state, deadline)
                if state.status == WorkflowStatus.FAILED:
                    raise StageFailure(state.errors[-1].code)
        except StageFailure:
            pass  # Terminal failure is returned as state and is recorded by the journal.
        return state

    def _fail(self, state: SessionState, code: str, recoverable: bool = False) -> None:
        state.errors.append(WorkflowError(code, state.current_step, recoverable))
        state.status = WorkflowStatus.FAILED
        state.current_step = Step.REPORT

    def _retry(self, state: SessionState, code: str, recoverable: bool) -> bool:
        state.errors.append(WorkflowError(code, state.current_step, recoverable))
        if recoverable and state.retry_count < self.config.max_retries:
            state.retry_count += 1
            state.reset_acquisition()
            state.current_step = Step.DATA_ACQUISITION
            return True
        state.status = WorkflowStatus.FAILED
        state.current_step = Step.REPORT
        return False

    def _run_steps(self, state: SessionState, deadline: float) -> None:
        for _ in range(self.config.max_workflow_steps):
            step = state.current_step
            # Reserve one invocation for a terminal report, including failures.
            if step != Step.REPORT and state.step_count >= self.config.max_workflow_steps - 1:
                self._fail(state, "MAX_WORKFLOW_STEPS")
                step = Step.REPORT
            if step != Step.REPORT and time.monotonic() >= deadline:
                self._fail(state, "WORKFLOW_DEADLINE")
                step = Step.REPORT
            agent = self.agents[step]
            task = AgentTask(state.request.session_id, step, state.retry_count)
            # A terminal report must remain available after a deadline expires.
            context = ToolContext(self.journal, agent.name, task,
                                  float("inf") if step == Step.REPORT else deadline, agent.tools)
            state.step_count += 1
            if step == Step.REPORT and state.status != WorkflowStatus.FAILED:
                state.status = WorkflowStatus.COMPLETED
            try:
                result = agent.execute(task, state, context)
            except Exception as exc:
                failure = exc if isinstance(exc, StageFailure) else StageFailure(f"{step.value}_{type(exc).__name__}")
                if step == Step.REPORT:
                    self._fail(state, failure.code)
                    return
                retryable = step == Step.DATA_ACQUISITION and failure.recoverable
                self._retry(state, failure.code, retryable)
                continue
            if step == Step.DATA_ACQUISITION:
                state.raw_data = result.payload
                state.current_step = Step.QUALITY_CHECK
            elif step == Step.QUALITY_CHECK:
                state.quality_report = result.payload
                if state.quality_report.status == QualityStatus.REJECTED:
                    self._retry(state, "SIGNAL_QUALITY_REJECTED", self.source.can_refresh)
                else:
                    state.current_step = Step.FEATURE_EXTRACTION
            elif step == Step.FEATURE_EXTRACTION:
                state.features = result.payload
                state.current_step = Step.PREDICTION
            elif step == Step.PREDICTION:
                state.prediction = result.payload
                state.current_step = Step.VERIFICATION
            elif step == Step.VERIFICATION:
                state.verification = result.payload
                state.current_step = Step.REPORT
            else:
                state.report = result.payload
                return
