"""Base interface for logging runs."""
from __future__ import annotations

from datetime import datetime

import datetime
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from langchain.callbacks.tracing.shared import (
    BaseTracer,
    ChainRun,
    LLMRun,
    ToolRun,
    TracerException,
)


@dataclass
class TracerStack(threading.local):
    """A stack of runs used for logging."""

    stack: List[Union[LLMRun, ChainRun, ToolRun]] = field(default_factory=list)
    execution_order: int = 1


class NestedTracer(Singleton, BaseTracer, ABC):
    """An implementation of the Tracer interface that prints trace as nested json."""

    _tracer_stack = TracerStack()

    def _start_trace(self, run: Union[LLMRun, ChainRun, ToolRun]) -> None:
        """Start a trace for a run."""

        self._tracer_stack.execution_order += 1

        if self._tracer_stack.stack:
            if not (
                isinstance(self._tracer_stack.stack[-1], ChainRun)
                or isinstance(self._tracer_stack.stack[-1], ToolRun)
            ):
                raise TracerException(
                    f"Nested {run.__class__.__name__} can only be logged inside a ChainRun or ToolRun"
                )
            self._add_child_run(self._tracer_stack.stack[-1], run)
        self._tracer_stack.stack.append(run)

    def _end_trace(self) -> None:
        """End a trace for a run."""

        run = self._tracer_stack.stack.pop()
        if not self._tracer_stack.stack:
            self._tracer_stack.execution_order = 1
            self._persist_run(run)

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
    def _generate_id(self) -> Optional[Union[int, str]]:
        """Generate an id for a run."""

    def start_llm_trace(
        self, serialized: Dict[str, Any], prompts: List[str], **extra: str
    ) -> None:
        """Start a trace for an LLM run."""

        llm_run = LLMRun(
            serialized=serialized,
            prompts={"prompts": prompts},
            extra=extra,
            start_time=datetime.datetime.utcnow(),
            error=None,
            execution_order=self._tracer_stack.execution_order,
            id=self._generate_id(),
            response=None,
            end_time=None,
        )
        self._start_trace(llm_run)

    def end_llm_trace(
        self, response: List[List[str]], error: Optional[str] = None
    ) -> None:
        """End a trace for an LLM run."""

        if not self._tracer_stack.stack or not isinstance(
            self._tracer_stack.stack[-1], LLMRun
        ):
            raise TracerException("No LLMRun found to be traced")

        self._tracer_stack.stack[-1].end_time = datetime.datetime.utcnow()
        self._tracer_stack.stack[-1].response = response
        self._tracer_stack.stack[-1].error = error

        self._end_trace()

    def start_chain_trace(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **extra: str
    ) -> None:
        """Start a trace for a chain run."""

        chain_run = ChainRun(
            serialized=serialized,
            inputs=inputs,
            extra=extra,
            start_time=datetime.datetime.utcnow(),
            error=None,
            execution_order=self._tracer_stack.execution_order,
            id=self._generate_id(),
            outputs=None,
            end_time=None,
            child_runs=[],
        )
        self._start_trace(chain_run)

    def end_chain_trace(
        self, outputs: Dict[str, Any], error: Optional[str] = None
    ) -> None:
        """End a trace for a chain run."""

        if not self._tracer_stack.stack or not isinstance(
            self._tracer_stack.stack[-1], ChainRun
        ):
            raise TracerException("No ChainRun found to be traced")

        self._tracer_stack.stack[-1].end_time = datetime.datetime.utcnow()
        self._tracer_stack.stack[-1].outputs = outputs
        self._tracer_stack.stack[-1].error = error

        self._end_trace()

    def start_tool_trace(
        self, serialized: Dict[str, Any], action: str, tool_input: str, **extra: str
    ) -> None:
        """Start a trace for a tool run."""

        tool_run = ToolRun(
            serialized=serialized,
            action=action,
            tool_input=tool_input,
            extra=extra,
            start_time=datetime.datetime.utcnow(),
            error=None,
            execution_order=self._tracer_stack.execution_order,
            id=self._generate_id(),
            output=None,
            end_time=None,
            child_runs=[],
        )
        self._start_trace(tool_run)

    def end_tool_trace(self, output: str, error: Optional[str] = None) -> None:
        """End a trace for a tool run."""

        if not self._tracer_stack.stack or not isinstance(
            self._tracer_stack.stack[-1], ToolRun
        ):
            raise TracerException("No ToolRun found to be traced")

        self._tracer_stack.stack[-1].end_time = datetime.datetime.utcnow()
        self._tracer_stack.stack[-1].output = output
        self._tracer_stack.stack[-1].error = error

        self._end_trace()
