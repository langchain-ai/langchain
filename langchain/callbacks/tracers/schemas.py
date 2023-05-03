"""Schemas for tracers."""
from __future__ import annotations

import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

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

    uuid: str
    parent_uuid: Optional[str] = None
    start_time: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    end_time: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    extra: Optional[Dict[str, Any]] = None
    execution_order: int
    child_execution_order: int
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


class ToolRun(BaseRun):
    """Class for ToolRun."""

    tool_input: str
    output: Optional[str] = None
    action: str
    child_llm_runs: List[LLMRun] = Field(default_factory=list)
    child_chain_runs: List[ChainRun] = Field(default_factory=list)
    child_tool_runs: List[ToolRun] = Field(default_factory=list)


class RunTypeEnum(str, Enum):
    """Enum for run types."""

    tool = "tool"
    chain = "chain"
    llm = "llm"


class Run(BaseModel):
    id: Optional[UUID]
    name: str
    start_time: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    end_time: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    extra: dict
    error: Optional[str]
    execution_order: int
    serialized: dict
    inputs: dict
    outputs: Optional[dict]
    session_id: int
    parent_run_id: Optional[UUID]
    example_id: Optional[UUID]
    run_type: RunTypeEnum
    child_runs: List[Run] = Field(default_factory=list)


ChainRun.update_forward_refs()
ToolRun.update_forward_refs()
