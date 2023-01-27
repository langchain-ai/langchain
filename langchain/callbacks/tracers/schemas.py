"""Schemas for tracers."""
from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from langchain.schema import LLMResult


class TracerSessionBase(BaseModel):
    """Base class for TracerSession."""

    start_time: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    name: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


class TracerSessionCreate(TracerSessionBase):
    """Create class for TracerSession."""

    pass


class TracerSession(TracerSessionBase):
    """TracerSession schema."""

    id: int


class BaseRun(BaseModel):
    """Base class for Run."""

    id: Optional[Union[int, str]] = None
    start_time: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    end_time: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    extra: Optional[Dict[str, Any]] = None
    execution_order: int
    serialized: Dict[str, Any]
    session_id: int
    error: Optional[str] = None


class LLMRun(BaseRun):
    """Class for LLMRun."""

    prompts: List[str]
    response: Optional[LLMResult] = None


class ChainRun(BaseRun):
    """Class for ChainRun."""

    inputs: Dict[str, Any]
    outputs: Optional[Dict[str, Any]] = None
    child_llm_runs: List[LLMRun] = Field(default_factory=list)
    child_chain_runs: List[ChainRun] = Field(default_factory=list)
    child_tool_runs: List[ToolRun] = Field(default_factory=list)
    child_runs: List[Union[LLMRun, ChainRun, ToolRun]] = Field(default_factory=list)


class ToolRun(BaseRun):
    """Class for ToolRun."""

    tool_input: str
    output: Optional[str] = None
    action: str
    child_llm_runs: List[LLMRun] = Field(default_factory=list)
    child_chain_runs: List[ChainRun] = Field(default_factory=list)
    child_tool_runs: List[ToolRun] = Field(default_factory=list)
    child_runs: List[Union[LLMRun, ChainRun, ToolRun]] = Field(default_factory=list)


ChainRun.update_forward_refs()
ToolRun.update_forward_refs()
