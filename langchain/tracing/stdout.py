"""An implementation of the Tracer interface that prints to stdout."""

from typing import Any, Dict, List, Optional, Union
from langchain.tracing.base import BaseTracer
from langchain.llms.base import LLMResult


class StdOutTracer(BaseTracer):
    """An implementation of the Tracer interface that prints to stdout."""

    def start_llm_trace(
        self, serialized: Dict[str, Any], prompts: List[str], **extra: str
    ) -> None:
        """Start a trace for an LLM run."""

        print(f"Starting LLM trace with prompts: {prompts}, serialized: {serialized}, extra: {extra}")

    def end_llm_trace(self, response: LLMResult, error=None) -> None:
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
