from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import time
from typing import Callable, Iterator, TypeVar

from src.agents.schemas import AgentTask
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)
T = TypeVar("T")


class StageFailure(Exception):
    """Only safe, static error codes cross the workflow/log boundary."""

    def __init__(self, code: str, recoverable: bool = False) -> None:
        self.code = code
        self.recoverable = recoverable
        super().__init__(code)


@dataclass(frozen=True)
class ExecutionEvent:
    run_id: str
    kind: str
    agent_name: str
    tool: str | None
    step: str
    retry_attempt: int
    start_time: str
    finish_time: str
    duration_ms: float
    status: str
    error: str | None
    tools_called: tuple[str, ...]


@dataclass(frozen=True)
class AgentStatistics:
    agent_name: str
    agent_calls: int
    tool_calls: int
    total_calls: int
    percentage: float


class ExecutionJournal:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.events: list[ExecutionEvent] = []

    @contextmanager
    def execution(self, agent: str, task: AgentTask, tool: str | None = None) -> Iterator[None]:
        start = datetime.now(timezone.utc).isoformat()
        clock = time.perf_counter()
        offset = len(self.events)
        status, error = "SUCCESS", None
        # Logs use a generated run ID; research participant/session IDs are not emitted.
        LOGGER.info("run=%s kind=%s agent=%s tool=%s step=%s retry=%s status=STARTED",
                    self.run_id, "tool" if tool else "agent", agent, tool, task.step.value, task.retry_attempt)
        try:
            yield
        except Exception as exc:
            status = "FAILED"
            error = exc.code if isinstance(exc, StageFailure) else type(exc).__name__
            raise
        finally:
            called = tuple(event.tool for event in self.events[offset:]
                           if event.kind == "tool" and event.agent_name == agent and event.tool is not None)
            event = ExecutionEvent(
                self.run_id, "tool" if tool else "agent", agent, tool, task.step.value, task.retry_attempt,
                start, datetime.now(timezone.utc).isoformat(), (time.perf_counter() - clock) * 1000,
                status, error, called,
            )
            self.events.append(event)
            LOGGER.info("execution=%s", asdict(event))

    def statistics(self) -> list[AgentStatistics]:
        names = sorted({event.agent_name for event in self.events})
        total = len(self.events)
        result = []
        for name in names:
            agents = sum(event.agent_name == name and event.kind == "agent" for event in self.events)
            tools = sum(event.agent_name == name and event.kind == "tool" for event in self.events)
            result.append(AgentStatistics(name, agents, tools, agents + tools, 100 * (agents + tools) / total))
        return result


class ToolContext:
    def __init__(self, journal: ExecutionJournal, agent: str, task: AgentTask, deadline: float,
                 allowed_tools: tuple[str, ...]) -> None:
        self.journal, self.agent, self.task, self.deadline = journal, agent, task, deadline
        if len(allowed_tools) > 5:
            raise ValueError("An agent may register at most five tools.")
        self.allowed_tools = allowed_tools

    def call(self, name: str, operation: Callable[..., T], *args: object) -> T:
        if name not in self.allowed_tools:
            raise StageFailure("UNREGISTERED_TOOL")
        with self.journal.execution(self.agent, self.task, tool=name):
            if time.monotonic() >= self.deadline:
                raise StageFailure("WORKFLOW_DEADLINE")
            result = operation(*args)
            if time.monotonic() >= self.deadline:
                raise StageFailure("WORKFLOW_DEADLINE")
            return result
