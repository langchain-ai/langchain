"""Base interface for logging runs."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List
from langchain.llms.base import LLMResult

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Run:
    id: int
    start_time: datetime
    end_time: datetime
    extra: Dict[str, Any]
    error: Dict[str, Any]
    execution_order: int
    serialized: Dict[str, Any]


@dataclass_json
@dataclass
class LLMRun(Run):
    prompts: Dict[str, Any]
    response: Dict[str, Any]


@dataclass_json
@dataclass
class ChainRun(Run):
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    child_runs: List[Run]


@dataclass_json
@dataclass
class ToolRun(Run):
    input: str
    output: str
    action: str
    child_runs: List[Run]


class BaseTracer(ABC):
    """Base interface for tracing runs."""

    @abstractmethod
    def start_llm_trace(
        self, serialized: Dict[str, Any], prompts: List[str], **extra: str
    ) -> None:
        """Start a trace for an LLM run."""

    @abstractmethod
    def end_llm_trace(self, response: LLMResult, error=None) -> None:
        """End a trace for an LLM run."""

    @abstractmethod
    def start_chain_trace(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **extra: str
    ) -> None:
        """Start a trace for a chain run."""

    @abstractmethod
    def end_chain_trace(self, outputs: Dict[str, Any], error=None) -> None:
        """End a trace for a chain run."""

    @abstractmethod
    def start_tool_trace(
        self,
        serialized: Dict[str, Any],
        action: str,
        inputs: str,
        **extra: str
    ) -> None:
        """Start a trace for a tool run."""

    @abstractmethod
    def end_tool_trace(self, output: str, error=None) -> None:
        """End a trace for a tool run."""
