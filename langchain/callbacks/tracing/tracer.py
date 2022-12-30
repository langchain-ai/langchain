from __future__ import annotations

import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from dataclasses_json import dataclass_json

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, LLMResult


@dataclass_json
@dataclass
class Run:
    id: Optional[Union[int, str]]
    start_time: datetime
    end_time: Optional[datetime]
    extra: Dict[str, Any]
    execution_order: int
    serialized: Dict[str, Any]


@dataclass_json
@dataclass
class LLMRun(Run):
    prompts: Dict[str, Any]
    response: Optional[List[List[str]]]


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
    """Base class for exceptions in tracing module."""


class BaseTracer(BaseCallbackHandler, ABC):
    """An implementation of the Tracer interface that prints trace as nested json."""

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

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Start a trace for an LLM run."""

        llm_run = LLMRun(
            serialized=serialized,
            prompts={"prompts": prompts},
            extra=kwargs,
            start_time=datetime.datetime.utcnow(),
            execution_order=self._tracer_stack.execution_order,
            id=self._generate_id(),
            response=None,
            end_time=None,
        )
        self._start_trace(llm_run)

    def on_llm_end(
        self,
        response: LLMResult,
    ) -> None:
        """End a trace for an LLM run."""

        if not self._tracer_stack.stack or not isinstance(
            self._tracer_stack.stack[-1], LLMRun
        ):
            raise TracerException("No LLMRun found to be traced")

        self._tracer_stack.stack[-1].end_time = datetime.datetime.utcnow()
        self._tracer_stack.stack[-1].response = response

        self._end_trace()

    def on_llm_error(self, error: Exception) -> None:
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Start a trace for a chain run."""

        chain_run = ChainRun(
            serialized=serialized,
            inputs=inputs,
            extra=kwargs,
            start_time=datetime.datetime.utcnow(),
            execution_order=self._tracer_stack.execution_order,
            id=self._generate_id(),
            outputs=None,
            end_time=None,
            child_runs=[],
        )
        self._start_trace(chain_run)

    def on_chain_end(self, outputs: Dict[str, Any]) -> None:
        """End a trace for a chain run."""

        if not self._tracer_stack.stack or not isinstance(
            self._tracer_stack.stack[-1], ChainRun
        ):
            raise TracerException("No ChainRun found to be traced")

        self._tracer_stack.stack[-1].end_time = datetime.datetime.utcnow()
        self._tracer_stack.stack[-1].outputs = outputs

        self._end_trace()

    def on_chain_error(self, error: Exception) -> None:
        pass

    def on_tool_start(
        self, serialized: Dict[str, Any], action: AgentAction, **kwargs: Any
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