"""Base interfaces for tracing runs."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.tracers.schemas import (
    ChainRun,
    LLMRun,
    ToolRun,
    TracerSession,
    TracerSessionCreate,
)
from langchain.schema import LLMResult


class TracerException(Exception):
    """Base class for exceptions in tracers module."""


class BaseTracer(BaseCallbackHandler, ABC):
    """Base interface for tracers."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.run_map: Dict[str, Union[LLMRun, ChainRun, ToolRun]] = {}
        self.execution_order: int = 1
        self.session: Optional[TracerSession] = None

    @staticmethod
    def _add_child_run(
        parent_run: Union[ChainRun, ToolRun],
        child_run: Union[LLMRun, ChainRun, ToolRun],
    ) -> None:
        """Add child run to a chain run or tool run."""
        if isinstance(child_run, LLMRun):
            parent_run.child_llm_runs.append(child_run)
        elif isinstance(child_run, ChainRun):
            parent_run.child_chain_runs.append(child_run)
        elif isinstance(child_run, ToolRun):
            parent_run.child_tool_runs.append(child_run)
        else:
            raise TracerException(f"Invalid run type: {type(child_run)}")

    @abstractmethod
    def _persist_run(self, run: Union[LLMRun, ChainRun, ToolRun]) -> None:
        """Persist a run."""

    @abstractmethod
    def _persist_session(self, session: TracerSessionCreate) -> TracerSession:
        """Persist a tracing session."""

    def new_session(self, name: Optional[str] = None, **kwargs: Any) -> TracerSession:
        """NOT thread safe, do not call this method from multiple threads."""
        session_create = TracerSessionCreate(name=name, extra=kwargs)
        session = self._persist_session(session_create)
        self.session = session
        return session

    @abstractmethod
    def load_session(self, session_name: str) -> TracerSession:
        """Load a tracing session and set it as the Tracer's session."""

    @abstractmethod
    def load_default_session(self) -> TracerSession:
        """Load the default tracing session and set it as the Tracer's session."""

    def _start_trace(self, run: Union[LLMRun, ChainRun, ToolRun]) -> None:
        """Start a trace for a run."""
        self.execution_order += 1

        if run.parent_uuid:
            parent_run = self.run_map[run.parent_uuid]
            if parent_run:
                if isinstance(parent_run, LLMRun):
                    raise TracerException(
                        "Cannot add child run to an LLM run. "
                        "LLM runs are not allowed to have children."
                    )
                self._add_child_run(parent_run, run)
            else:
                raise TracerException(
                    f"Parent run with UUID {run.parent_uuid} not found."
                )

        self.run_map[run.uuid] = run

    def _end_trace(self, run) -> None:
        """End a trace for a run."""
        if not run.parent_uuid:
            self._persist_run(run)
            self.execution_order = 1
        self.run_map.pop(run.uuid)

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        run_id: str = None,
        parent_run_id: str = None,
        **kwargs: Any,
    ) -> None:
        """Start a trace for an LLM run."""
        if self.session is None:
            self.session = self.load_default_session()

        if run_id is None:
            run_id = str(uuid4())

        llm_run = LLMRun(
            uuid=run_id,
            parent_uuid=parent_run_id,
            serialized=serialized,
            prompts=prompts,
            extra=kwargs,
            start_time=datetime.utcnow(),
            execution_order=self.execution_order,
            session_id=self.session.id,
        )
        self._start_trace(llm_run)

    def on_llm_end(
        self, response: LLMResult, run_id: str = None, **kwargs: Any
    ) -> None:
        """End a trace for an LLM run."""
        if not run_id:
            raise TracerException("No run_id provided for on_llm_end callback.")

        if not self.run_map or not isinstance(self.run_map[run_id], LLMRun):
            raise TracerException("No LLMRun found to be traced")

        llm_run = self.run_map[run_id]
        llm_run.response = response
        llm_run.end_time = datetime.utcnow()
        self._end_trace(llm_run)

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        run_id: str = None,
        **kwargs: Any,
    ) -> None:
        """Handle an error for an LLM run."""
        if not run_id:
            raise TracerException("No run_id provided for on_llm_error callback.")

        if not self.run_map or not isinstance(self.run_map[run_id], LLMRun):
            raise TracerException("No LLMRun found to be traced")

        llm_run = self.run_map[run_id]
        llm_run.error = repr(error)
        llm_run.end_time = datetime.utcnow()
        self._end_trace(llm_run)

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        run_id: str = None,
        parent_run_id: str = None,
        **kwargs: Any,
    ) -> None:
        """Start a trace for a chain run."""
        if self.session is None:
            self.session = self.load_default_session()

        if run_id is None:
            run_id = str(uuid4())

        chain_run = ChainRun(
            uuid=run_id,
            parent_uuid=parent_run_id,
            serialized=serialized,
            inputs=inputs,
            extra=kwargs,
            start_time=datetime.utcnow(),
            execution_order=self.execution_order,
            child_runs=[],
            session_id=self.session.id,
        )
        self._start_trace(chain_run)

    def on_chain_end(
        self, outputs: Dict[str, Any], run_id: str = None, **kwargs: Any
    ) -> None:
        """End a trace for a chain run."""
        if not run_id:
            raise TracerException("No run_id provided for on_chain_end callback.")

        if not self.run_map or not isinstance(self.run_map[run_id], ChainRun):
            raise TracerException("No ChainRun found to be traced")

        chain_run = self.run_map[run_id]
        chain_run.outputs = outputs
        chain_run.end_time = datetime.utcnow()
        self._end_trace(chain_run)

    def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        run_id: str = None,
        **kwargs: Any,
    ) -> None:
        """Handle an error for a chain run."""
        if not run_id:
            raise TracerException("No run_id provided for on_chain_error callback.")

        if not self.run_map or not isinstance(self.run_map[run_id], ChainRun):
            raise TracerException("No ChainRun found to be traced")

        chain_run = self.run_map[run_id]
        chain_run.error = repr(error)
        chain_run.end_time = datetime.utcnow()
        self._end_trace(chain_run)

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        run_id: str = None,
        parent_run_id: str = None,
        **kwargs: Any,
    ) -> None:
        """Start a trace for a tool run."""
        if self.session is None:
            self.session = self.load_default_session()

        if run_id is None:
            run_id = str(uuid4())

        tool_run = ToolRun(
            uuid=run_id,
            parent_uuid=parent_run_id,
            serialized=serialized,
            # TODO: this is duplicate info as above, not needed.
            action=str(serialized),
            tool_input=input_str,
            extra=kwargs,
            start_time=datetime.utcnow(),
            execution_order=self.execution_order,
            child_runs=[],
            session_id=self.session.id,
        )
        self._start_trace(tool_run)

    def on_tool_end(self, output: str, run_id: str = None, **kwargs: Any) -> None:
        """End a trace for a tool run."""
        if not run_id:
            raise TracerException("No run_id provided for on_tool_end callback.")

        if not self.run_map or not isinstance(self.run_map[run_id], ToolRun):
            raise TracerException("No ToolRun found to be traced")

        tool_run = self.run_map[run_id]
        tool_run.output = output
        tool_run.end_time = datetime.utcnow()
        self._end_trace(tool_run)

    def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        run_id: str = None,
        **kwargs: Any,
    ) -> None:
        """Handle an error for a tool run."""
        if not run_id:
            raise TracerException("No run_id provided for on_tool_error callback.")

        if not self.run_map or not isinstance(self.run_map[run_id], ToolRun):
            raise TracerException("No ToolRun found to be traced")

        tool_run = self.run_map[run_id]
        tool_run.error = repr(error)
        tool_run.end_time = datetime.utcnow()
        self._end_trace(tool_run)
