"""An implementation of the Tracer interface that prints trace as nested json."""

from typing import Any, Dict, List, Optional, Union
from langchain.tracing.base import BaseTracer, ChainRun, LLMRun, ToolRun, TracerException


class NestedJsonTracer(BaseTracer):
    """An implementation of the Tracer interface that prints trace as nested json."""

    def __init__(self):
        self._stack = []
        self._execution_order = 1

    def _log_run_start(self, run: Union[LLMRun, ChainRun, ToolRun]) -> None:
        """Log the start of a run."""

        run.execution_order = self._execution_order
        self._execution_order += 1

        if len(session.stack):
            if not (
                    isinstance(session.stack[-1], ChainRun)
                    or isinstance(session.stack[-1], ToolRun)
            ):
                session.rollback()
                raise LoggerException(
                    f"Nested {run.__class__.__name__} can only be logged inside a ChainRun or ToolRun"
                )
            if isinstance(run, LLMRun):
                session.stack[-1].child_llm_runs.append(run)
            elif isinstance(run, ChainRun):
                session.stack[-1].child_chain_runs.append(run)
            else:
                session.stack[-1].child_tool_runs.append(run)
        session.stack.append(run)
        run.save()

    def start_llm_trace(
        self, serialized: Dict[str, Any], prompts: List[str], **extra: str
    ) -> None:
        """Start a trace for an LLM run."""

        print(f"Starting LLM trace with prompts: {prompts}, serialized: {serialized}, extra: {extra}")

    def end_llm_trace(self, response: List[List[str]], error=None) -> None:
        """End a trace for an LLM run."""

        print(f"Ending LLM trace with response: {response}, error: {error}")

    def start_chain_trace(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **extra: str
    ) -> None:
        """Start a trace for a chain run."""

        print(f"Starting chain trace with inputs: {inputs}, serialized: {serialized}, extra: {extra}")

    def end_chain_trace(self, outputs: Dict[str, Any], error=None) -> None:
        """End a trace for a chain run."""

        print(f"Ending chain trace with outputs: {outputs}, error: {error}")

    def start_tool_trace(
        self,
        serialized: Dict[str, Any],
        action: str,
        inputs: str,
        **extra: str
    ) -> None:
        """Start a trace for a tool run."""

        print(f"Starting tool trace with inputs: {inputs}, serialized: {serialized}, extra: {extra}")

    def end_tool_trace(self, output: str, error=None) -> None:
        """End a trace for a tool run."""

        print(f"Ending tool trace with output: {output}, error: {error}")