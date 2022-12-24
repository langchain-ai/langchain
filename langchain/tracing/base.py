"""Base interface for logging runs."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Run:
    id: Union[int, str]
    start_time: datetime
    end_time: Optional[datetime]
    extra: Dict[str, Any]
    error: Optional[str]
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
    child_runs: List[Run]  # Consolidated child runs

    child_llm_runs: List[LLMRun]
    child_chain_runs: List[ChainRun]
    child_tool_runs: List[ToolRun]


@dataclass_json
@dataclass
class ToolRun(Run):
    tool_input: str
    output: Optional[str]
    action: str
    child_runs: List[Run]  # Consolidated child runs

    child_llm_runs: List[LLMRun]
    child_chain_runs: List[ChainRun]
    child_tool_runs: List[ToolRun]


class TracerException(Exception):
    """Base class for exceptions in tracing module."""


class BaseTracer(ABC):
    """Base interface for tracing runs."""

    @abstractmethod
    def start_llm_trace(
        self, serialized: Dict[str, Any], prompts: List[str], **extra: str
    ) -> None:
        """Start a trace for an LLM run."""

    @abstractmethod
    def end_llm_trace(
        self, response: List[List[str]], error: Optional[str] = None
    ) -> None:
        """End a trace for an LLM run."""

    @abstractmethod
    def start_chain_trace(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **extra: str
    ) -> None:
        """Start a trace for a chain run."""

    @abstractmethod
    def end_chain_trace(
        self, outputs: Dict[str, Any], error: Optional[str] = None
    ) -> None:
        """End a trace for a chain run."""

    @abstractmethod
    def start_tool_trace(
        self, serialized: Dict[str, Any], action: str, tool_input: str, **extra: str
    ) -> None:
        """Start a trace for a tool run."""

    @abstractmethod
    def end_tool_trace(self, output: str, error: Optional[str] = None) -> None:
        """End a trace for a tool run."""
