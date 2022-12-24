"""An implementation of the Tracer interface that POSTS to the langchain endpoint."""

import datetime
from typing import Any, Dict, List, Optional, Union

import requests

from langchain.tracing.base import (
    BaseTracer,
    ChainRun,
    LLMRun,
    Run,
    ToolRun,
    TracerException,
)


class LangChainTracer(BaseTracer):
    """An implementation of the Tracer interface that POSTS to the langchain endpoint."""

    _instance = None
    _stack: List[Run] = None
    _execution_order: int = None
    _endpoint: str = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "LangChainTracer":
        if cls._instance is None:
            cls._instance = super(LangChainTracer, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._stack is None:
            self._stack = []
        if self._execution_order is None:
            self._execution_order = 1
        if self._endpoint is None:
            self._endpoint = "http://127.0.0.1:5000"

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
            if isinstance(run, LLMRun):
                self._stack[-1].child_llm_runs.append(run)
            elif isinstance(run, ChainRun):
                self._stack[-1].child_chain_runs.append(run)
            else:
                self._stack[-1].child_tool_runs.append(run)
        self._stack.append(run)

    def _end_trace(self) -> None:
        """End a trace for a run."""

        run = self._stack.pop()
        if not self._stack:
            self._execution_order = 1
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
            print(
                f"POST {endpoint}, status code: {r.status_code}, id: {r.json()['id']}"
            )

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
            execution_order=self._execution_order,
            response=None,
            end_time=None,
            id=None,
        )
        self._start_trace(llm_run)

    def end_llm_trace(
        self, response: List[List[str]], error: Optional[str] = None
    ) -> None:
        """End a trace for an LLM run."""

        if not self._stack or not isinstance(self._stack[-1], LLMRun):
            raise TracerException("No LLMRun found to be traced")

        self._stack[-1].end_time = datetime.datetime.utcnow()
        self._stack[-1].response = response
        self._stack[-1].error = error

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
            execution_order=self._execution_order,
            outputs=None,
            end_time=None,
            child_runs=[],
            id=None,
        )
        self._start_trace(chain_run)

    def end_chain_trace(
        self, outputs: Dict[str, Any], error: Optional[str] = None
    ) -> None:
        """End a trace for a chain run."""

        if not self._stack or not isinstance(self._stack[-1], ChainRun):
            raise TracerException("No ChainRun found to be traced")

        self._stack[-1].end_time = datetime.datetime.utcnow()
        self._stack[-1].outputs = outputs
        self._stack[-1].error = error

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
            execution_order=self._execution_order,
            output=None,
            end_time=None,
            child_runs=[],
            id=None,
        )
        self._start_trace(tool_run)

    def end_tool_trace(self, output: str, error: Optional[str] = None) -> None:
        """End a trace for a tool run."""

        if not self._stack or not isinstance(self._stack[-1], ToolRun):
            raise TracerException("No ToolRun found to be traced")

        self._stack[-1].end_time = datetime.datetime.utcnow()
        self._stack[-1].output = output
        self._stack[-1].error = error

        self._end_trace()
