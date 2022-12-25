"""An implementation of the Tracer interface that prints trace as nested json."""

import datetime
import uuid
from typing import Any, Dict, List, Optional, Union
import threading
from dataclasses import dataclass, field

from langchain.tracing.base import (
    BaseTracer,
    ChainRun,
    LLMRun,
    Run,
    ToolRun,
    TracerException,
)


class Singleton:
    """A thread-safe singleton class that can be inherited from."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                # Another thread could have created the instance
                # before we acquired the lock. So check that the
                # instance is still nonexistent.
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance


@dataclass
class TracerStack(threading.local):
    """A stack of runs used for logging."""

    stack: List[Union[LLMRun, ChainRun, ToolRun]] = field(default_factory=list)
    execution_order: int = 1


class JsonTracer(Singleton, BaseTracer):
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
            self._tracer_stack.stack[-1].child_runs.append(run)
        self._tracer_stack.stack.append(run)

    def _end_trace(self) -> None:
        """End a trace for a run."""

        run = self._tracer_stack.stack.pop()
        if not self._tracer_stack.stack:
            self._tracer_stack.execution_order = 1
            print(run.to_json(indent=2))

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
            id=str(uuid.uuid4()),
            response=None,
            end_time=None,
        )
        self._start_trace(llm_run)

    def end_llm_trace(
        self, response: List[List[str]], error: Optional[str] = None
    ) -> None:
        """End a trace for an LLM run."""

        if not self._tracer_stack.stack or not isinstance(self._tracer_stack.stack[-1], LLMRun):
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
            id=str(uuid.uuid4()),
            outputs=None,
            end_time=None,
            child_runs=[],
        )
        self._start_trace(chain_run)

    def end_chain_trace(
        self, outputs: Dict[str, Any], error: Optional[str] = None
    ) -> None:
        """End a trace for a chain run."""

        if not self._tracer_stack.stack or not isinstance(self._tracer_stack.stack[-1], ChainRun):
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
            id=str(uuid.uuid4()),
            output=None,
            end_time=None,
            child_runs=[],
        )
        self._start_trace(tool_run)

    def end_tool_trace(self, output: str, error: Optional[str] = None) -> None:
        """End a trace for a tool run."""

        if not self._tracer_stack.stack or not isinstance(self._tracer_stack.stack[-1], ToolRun):
            raise TracerException("No ToolRun found to be traced")

        self._tracer_stack.stack[-1].end_time = datetime.datetime.utcnow()
        self._tracer_stack.stack[-1].output = output
        self._tracer_stack.stack[-1].error = error

        self._end_trace()
