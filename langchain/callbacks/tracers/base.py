"""Base interfaces for tracing runs."""
from __future__ import annotations

import datetime
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import requests
from dataclasses_json import dataclass_json
from pydantic import BaseModel

from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.shared import Singleton
from langchain.schema import AgentAction, LLMResult


@dataclass_json
@dataclass
class TracerSession:
    id: Optional[Union[int, str]]
    start_time: datetime = field(default_factory=datetime.utcnow)
    extra: Dict[str, Any] = field(default_factory=dict)
    child_runs: List[Run] = field(default_factory=list)  # Consolidated child runs


@dataclass_json
@dataclass
class Run:
    id: Optional[Union[int, str]]
    start_time: datetime
    end_time: Optional[datetime]
    extra: Dict[str, Any]
    execution_order: int
    serialized: Dict[str, Any]
    session_id: Optional[Union[int, str]]


@dataclass_json
@dataclass
class LLMRun(Run):
    prompts: List[str]
    response: Optional[LLMResult]


@dataclass_json
@dataclass
class ChainRun(Run):
    inputs: Dict[str, Any]
    outputs: Optional[Dict[str, Any]]
    child_runs: List[Run] = field(default_factory=list)  # Consolidated child runs

    child_llm_runs: List[LLMRun] = field(default_factory=list)
    child_chain_runs: List[ChainRun] = field(default_factory=list)
    child_tool_runs: List[ToolRun] = field(default_factory=list)


@dataclass_json
@dataclass
class ToolRun(Run):
    tool_input: str
    output: Optional[str]
    action: str
    child_runs: List[Run] = field(default_factory=list)  # Consolidated child runs

    child_llm_runs: List[LLMRun] = field(default_factory=list)
    child_chain_runs: List[ChainRun] = field(default_factory=list)
    child_tool_runs: List[ToolRun] = field(default_factory=list)


class TracerException(Exception):
    """Base class for exceptions in tracers module."""


class BaseTracer(BaseCallbackHandler, ABC):
    """An implementation of the Tracer interface that prints trace as nested json."""

    @abstractmethod
    def _add_child_run(
        self,
        parent_run: Union[ChainRun, ToolRun],
        child_run: Union[LLMRun, ChainRun, ToolRun],
    ) -> None:
        """Add child run to a chain run or tool run."""

    @abstractmethod
    def _persist_run(self, run: Union[LLMRun, ChainRun, ToolRun]) -> None:
        """Persist a run."""

    @abstractmethod
    def _persist_session(self, session: TracerSession) -> None:
        """Persist a tracing session."""

    @abstractmethod
    def _generate_id(self) -> Optional[Union[int, str]]:
        """Generate an id for a run."""

    def new_session(self, **kwargs) -> TracerSession:
        """Start a new tracing session. NOT thread safe, do not call this method from multiple threads."""
        session = TracerSession(id=None, extra=kwargs)
        self._persist_session(session)
        self._session = session
        return session

    @property
    @abstractmethod
    def _stack(self) -> List[Union[LLMRun, ChainRun, ToolRun]]:
        """Get the tracer stack."""

    @property
    @abstractmethod
    def _execution_order(self) -> int:
        """Get the execution order for a run."""

    @_execution_order.setter
    @abstractmethod
    def _execution_order(self, value: int) -> None:
        """Set the execution order for a run."""

    @property
    @abstractmethod
    def _session(self) -> Optional[TracerSession]:
        """Get the tracing session."""

    @_session.setter
    @abstractmethod
    def _session(self, value: TracerSession) -> None:
        """Set the tracing session."""

    def _start_trace(self, run: Union[LLMRun, ChainRun, ToolRun]) -> None:
        """Start a trace for a run."""

        self._execution_order += 1

        if self._stack:
            if not (
                isinstance(self._stack[-1], ChainRun)
                or isinstance(self._stack[-1], ToolRun)
            ):
                raise TracerException(
                    f"Nested {run.__class__.__name__} can only be logged inside a ChainRun or ToolRun"
                )
            self._add_child_run(self._stack[-1], run)
        self._stack.append(run)

    def _end_trace(self) -> None:
        """End a trace for a run."""

        run = self._stack.pop()
        if not self._stack:
            self._session.child_runs.append(run)
            self._execution_order = 1
            self._persist_run(run)

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Start a trace for an LLM run."""

        if self._session is None:
            raise TracerException("Initialize a session with `new_session()` before starting a trace.")

        llm_run = LLMRun(
            serialized=serialized,
            prompts=prompts,
            extra=kwargs,
            start_time=datetime.utcnow(),
            execution_order=self._execution_order,
            id=self._generate_id(),
            response=None,
            end_time=None,
            session_id=self._session.id,
        )
        self._start_trace(llm_run)

    def on_llm_end(
        self,
        response: LLMResult,
    ) -> None:
        """End a trace for an LLM run."""

        if not self._stack or not isinstance(self._stack[-1], LLMRun):
            raise TracerException("No LLMRun found to be traced")

        self._stack[-1].end_time = datetime.utcnow()
        self._stack[-1].response = response

        self._end_trace()

    def on_llm_error(self, error: Exception) -> None:
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Start a trace for a chain run."""

        if self._session is None:
            raise TracerException("Initialize a session with `new_session()` before starting a trace.")

        chain_run = ChainRun(
            serialized=serialized,
            inputs=inputs,
            extra=kwargs,
            start_time=datetime.utcnow(),
            execution_order=self._execution_order,
            id=self._generate_id(),
            outputs=None,
            end_time=None,
            child_runs=[],
            session_id=self._session.id,
        )
        self._start_trace(chain_run)

    def on_chain_end(self, outputs: Dict[str, Any]) -> None:
        """End a trace for a chain run."""

        if not self._stack or not isinstance(self._stack[-1], ChainRun):
            raise TracerException("No ChainRun found to be traced")

        self._stack[-1].end_time = datetime.utcnow()
        self._stack[-1].outputs = outputs

        self._end_trace()

    def on_chain_error(self, error: Exception) -> None:
        pass

    def on_tool_start(
        self, serialized: Dict[str, Any], action: AgentAction, **kwargs: Any
    ) -> None:
        """Start a trace for a tool run."""

        if self._session is None:
            raise TracerException("Initialize a session with `new_session()` before starting a trace.")

        tool_run = ToolRun(
            serialized=serialized,
            action=action.tool,
            tool_input=action.tool_input,
            extra=kwargs,
            start_time=datetime.utcnow(),
            execution_order=self._execution_order,
            id=self._generate_id(),
            output=None,
            end_time=None,
            child_runs=[],
            session_id=self._session.id,
        )
        self._start_trace(tool_run)

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """End a trace for a tool run."""

        if not self._stack or not isinstance(self._stack[-1], ToolRun):
            raise TracerException("No ToolRun found to be traced")

        self._stack[-1].end_time = datetime.utcnow()
        self._stack[-1].output = output

        self._end_trace()

    def on_tool_error(self, error: Exception) -> None:
        pass

    def on_text(self, text: str, **kwargs: Any) -> None:
        pass


class Tracer(BaseTracer, ABC):
    """A non-thread safe implementation of the BaseTracer interface."""

    def __init__(self) -> None:
        """Initialize a tracer."""
        self._tracer_stack: List[Union[LLMRun, ChainRun, ToolRun]] = []
        self._tracer_execution_order = 1
        self._tracer_session = None

    @property
    def _stack(self) -> List[Union[LLMRun, ChainRun, ToolRun]]:
        """Get the tracer stack."""
        return self._tracer_stack

    @property
    def _execution_order(self) -> int:
        """Get the execution order for a run."""
        return self._tracer_execution_order

    @_execution_order.setter
    def _execution_order(self, value: int) -> None:
        """Set the execution order for a run."""
        self._tracer_execution_order = value

    @property
    def _session(self) -> Optional[TracerSession]:
        """Get the tracing session."""
        return self._tracer_session

    @_session.setter
    def _session(self, value: TracerSession) -> None:
        """Set the tracing session."""

        if self._stack:
            raise TracerException("Cannot set a session while a trace is being recorded")
        self._tracer_session = value


@dataclass
class TracerStack(threading.local):
    """A stack of runs used for logging."""

    stack: List[Union[LLMRun, ChainRun, ToolRun]] = field(default_factory=list)
    execution_order: int = 1


class SharedTracer(Singleton, BaseTracer, ABC):
    """A thread-safe Singleton implementation of BaseTracer."""

    _tracer_stack = TracerStack()

    @property
    def _stack(self) -> List[Union[LLMRun, ChainRun, ToolRun]]:
        """Get the tracer stack."""
        return self._tracer_stack.stack

    @property
    def _execution_order(self) -> int:
        """Get the execution order for a run."""
        return self._tracer_stack.execution_order

    @_execution_order.setter
    def _execution_order(self, value: int) -> None:
        """Set the execution order for a run."""
        self._tracer_stack.execution_order = value

    @property
    def _session(self) -> Optional[TracerSession]:
        """Get the tracing session."""
        return self._tracer_session

    @_session.setter
    def _session(self, value: TracerSession) -> None:
        """Set the tracing session."""
        with self._lock:
            # TODO: currently, we are only checking current thread's stack. Need to make sure that
            #   we are not in the middle of a trace in any thread.
            if self._stack:
                raise TracerException("Cannot set a session while a trace is being recorded")
            self._tracer_session = value


class BaseJsonTracer(BaseTracer, ABC):
    """An implementation of SharedTracer that prints trace as nested json."""

    def _persist_run(self, run: Union[LLMRun, ChainRun, ToolRun]) -> None:
        """Persist a run."""

        print(run.to_json(indent=2))

    def _persist_session(self, session: TracerSession) -> None:
        """Persist a session."""

        print(session.to_json(indent=2))

    def _add_child_run(
        self,
        parent_run: Union[ChainRun, ToolRun],
        child_run: Union[LLMRun, ChainRun, ToolRun],
    ) -> None:
        """Add child run to a chain run or tool run."""

        parent_run.child_runs.append(child_run)

    def _generate_id(self) -> Optional[Union[int, str]]:
        """Generate an id for a run."""

        return str(uuid.uuid4())


class BaseLangChainTracer(BaseTracer, ABC):
    """An implementation of the SharedTracer that POSTS to the langchain endpoint."""

    _endpoint: str = "http://127.0.0.1:5000"

    def _persist_run(self, run: Union[LLMRun, ChainRun, ToolRun]) -> None:
        """Persist a run."""

        if isinstance(run, LLMRun):
            endpoint = f"{self._endpoint}/llm-runs"
        elif isinstance(run, ChainRun):
            endpoint = f"{self._endpoint}/chain-runs"
        else:
            endpoint = f"{self._endpoint}/tool-runs"
        r = requests.post(
            endpoint,
            data=run.to_json(),
            headers={"Content-Type": "application/json"},
        )
        print(f"POST {endpoint}, status code: {r.status_code}, id: {r.json()['id']}")

    def _persist_session(self, session: TracerSession) -> None:
        """Persist a session."""

        r = requests.post(
            f"{self._endpoint}/sessions",
            data=session.to_json(),
            headers={"Content-Type": "application/json"},
        )
        print(f"POST {self._endpoint}/sessions, status code: {r.status_code}, id: {r.json()['id']}")
        session.id = r.json()["id"]

    def _add_child_run(
        self,
        parent_run: Union[ChainRun, ToolRun],
        child_run: Union[LLMRun, ChainRun, ToolRun],
    ) -> None:
        """Add child run to a chain run or tool run."""

        if isinstance(child_run, LLMRun):
            parent_run.child_llm_runs.append(child_run)
        elif isinstance(child_run, ChainRun):
            parent_run.child_chain_runs.append(child_run)
        else:
            parent_run.child_tool_runs.append(child_run)

    def _generate_id(self) -> Optional[Union[int, str]]:
        """Generate an id for a run."""

        return None
