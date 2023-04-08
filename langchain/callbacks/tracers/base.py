"""Base interfaces for tracing runs."""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.shared import Singleton
from langchain.callbacks.tracers.schemas import (
    ChainRun,
    LLMRun,
    ToolRun,
    TracerSession,
    TracerSessionCreate,
)
from langchain.schema import AgentAction, AgentFinish, LLMResult


class TracerException(Exception):
    """Base class for exceptions in tracers module."""


class BaseTracer(BaseCallbackHandler, ABC):
    """Base interface for tracers."""

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
    def _persist_session(self, session: TracerSessionCreate) -> TracerSession:
        """Persist a tracing session."""

    @abstractmethod
    def _generate_id(self) -> Optional[Union[int, str]]:
        """Generate an id for a run."""

    def new_session(self, name: Optional[str] = None, **kwargs: Any) -> TracerSession:
        """NOT thread safe, do not call this method from multiple threads."""
        session_create = TracerSessionCreate(name=name, extra=kwargs)
        session = self._persist_session(session_create)
        self._session = session
        return session

    @abstractmethod
    def load_session(self, session_name: str) -> TracerSession:
        """Load a tracing session and set it as the Tracer's session."""

    @abstractmethod
    def load_default_session(self) -> TracerSession:
        """Load the default tracing session and set it as the Tracer's session."""

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
                    f"Nested {run.__class__.__name__} can only be"
                    f" logged inside a ChainRun or ToolRun"
                )
            self._add_child_run(self._stack[-1], run)
        self._stack.append(run)

    def _end_trace(self) -> None:
        """End a trace for a run."""
        run = self._stack.pop()
        if not self._stack:
            self._execution_order = 1
            self._persist_run(run)

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Start a trace for an LLM run."""
        if self._session is None:
            raise TracerException(
                "Initialize a session with `new_session()` before starting a trace."
            )

        llm_run = LLMRun(
            serialized=serialized,
            prompts=prompts,
            extra=kwargs,
            start_time=datetime.utcnow(),
            execution_order=self._execution_order,
            session_id=self._session.id,
            id=self._generate_id(),
        )
        self._start_trace(llm_run)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Handle a new token for an LLM run."""
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """End a trace for an LLM run."""
        if not self._stack or not isinstance(self._stack[-1], LLMRun):
            raise TracerException("No LLMRun found to be traced")

        self._stack[-1].end_time = datetime.utcnow()
        self._stack[-1].response = response

        self._end_trace()

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Handle an error for an LLM run."""
        if not self._stack or not isinstance(self._stack[-1], LLMRun):
            raise TracerException("No LLMRun found to be traced")

        self._stack[-1].error = repr(error)
        self._stack[-1].end_time = datetime.utcnow()

        self._end_trace()

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Start a trace for a chain run."""
        if self._session is None:
            raise TracerException(
                "Initialize a session with `new_session()` before starting a trace."
            )

        chain_run = ChainRun(
            serialized=serialized,
            inputs=inputs,
            extra=kwargs,
            start_time=datetime.utcnow(),
            execution_order=self._execution_order,
            child_runs=[],
            session_id=self._session.id,
            id=self._generate_id(),
        )
        self._start_trace(chain_run)

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """End a trace for a chain run."""
        if not self._stack or not isinstance(self._stack[-1], ChainRun):
            raise TracerException("No ChainRun found to be traced")

        self._stack[-1].end_time = datetime.utcnow()
        self._stack[-1].outputs = outputs

        self._end_trace()

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Handle an error for a chain run."""
        if not self._stack or not isinstance(self._stack[-1], ChainRun):
            raise TracerException("No ChainRun found to be traced")

        self._stack[-1].end_time = datetime.utcnow()
        self._stack[-1].error = repr(error)

        self._end_trace()

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Start a trace for a tool run."""
        if self._session is None:
            raise TracerException(
                "Initialize a session with `new_session()` before starting a trace."
            )

        tool_run = ToolRun(
            serialized=serialized,
            # TODO: this is duplicate info as above, not needed.
            action=str(serialized),
            tool_input=input_str,
            extra=kwargs,
            start_time=datetime.utcnow(),
            execution_order=self._execution_order,
            child_runs=[],
            session_id=self._session.id,
            id=self._generate_id(),
        )
        self._start_trace(tool_run)

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """End a trace for a tool run."""
        if not self._stack or not isinstance(self._stack[-1], ToolRun):
            raise TracerException("No ToolRun found to be traced")

        self._stack[-1].end_time = datetime.utcnow()
        self._stack[-1].output = output

        self._end_trace()

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Handle an error for a tool run."""
        if not self._stack or not isinstance(self._stack[-1], ToolRun):
            raise TracerException("No ToolRun found to be traced")

        self._stack[-1].end_time = datetime.utcnow()
        self._stack[-1].error = repr(error)

        self._end_trace()

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Handle a text message."""
        pass

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Handle an agent finish message."""
        pass

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Do nothing."""
        pass


class Tracer(BaseTracer, ABC):
    """A non-thread safe implementation of the BaseTracer interface."""

    def __init__(self) -> None:
        """Initialize a tracer."""
        self._tracer_stack: List[Union[LLMRun, ChainRun, ToolRun]] = []
        self._tracer_execution_order = 1
        self._tracer_session: Optional[TracerSession] = None

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
            raise TracerException(
                "Cannot set a session while a trace is being recorded"
            )
        self._tracer_session = value


@dataclass
class TracerStack(threading.local):
    """A stack of runs used for logging."""

    stack: List[Union[LLMRun, ChainRun, ToolRun]] = field(default_factory=list)
    execution_order: int = 1


class SharedTracer(Singleton, BaseTracer, ABC):
    """A thread-safe Singleton implementation of BaseTracer."""

    _tracer_stack = TracerStack()
    _tracer_session = None

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
            # TODO: currently, we are only checking current thread's stack.
            #  Need to make sure that we are not in the middle of a trace
            #  in any thread.
            if self._stack:
                raise TracerException(
                    "Cannot set a session while a trace is being recorded"
                )
            self._tracer_session = value
