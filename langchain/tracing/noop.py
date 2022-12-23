"""An implementation of the Tracer interface that does nothing."""

from typing import Any, Dict, List, Optional

from langchain.tracing.base import BaseTracer


class NoOpTracer(BaseTracer):
    """An implementation of the Tracer interface that does nothing."""

    def start_llm_trace(
        self, serialized: Dict[str, Any], prompts: List[str], **extra: str
    ) -> None:
        """Start a trace for an LLM run."""

    def end_llm_trace(
        self, response: List[List[str]], error: Optional[str] = None
    ) -> None:
        """End a trace for an LLM run."""

    def start_chain_trace(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **extra: str
    ) -> None:
        """Start a trace for a chain run."""

    def end_chain_trace(
        self, outputs: Dict[str, Any], error: Optional[str] = None
    ) -> None:
        """End a trace for a chain run."""

    def start_tool_trace(
        self, serialized: Dict[str, Any], action: str, tool_input: str, **extra: str
    ) -> None:
        """Start a trace for a tool run."""

    def end_tool_trace(self, output: str, error: Optional[str] = None) -> None:
        """End a trace for a tool run."""
