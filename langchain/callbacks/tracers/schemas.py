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


class TracerSessionV2Base(TracerSessionBase):
    """A creation class for TracerSessionV2."""

    tenant_id: UUID


class TracerSessionV2Create(TracerSessionV2Base):
    """A creation class for TracerSessionV2."""

    id: Optional[UUID]

    pass


class TracerSessionV2(TracerSessionV2Base):
    """TracerSession schema for the V2 API."""

    id: UUID


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


class RunBase(BaseModel):
    """Base Run schema."""

    id: Optional[UUID]
    start_time: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    end_time: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    extra: dict
    error: Optional[str]
    execution_order: int
    serialized: dict
    inputs: dict
    outputs: Optional[dict]
    session_id: UUID
    reference_example_id: Optional[UUID]
    run_type: RunTypeEnum


class RunCreate(RunBase):
    """Schema to create a run in the DB."""

    name: Optional[str]
    child_runs: List[RunCreate] = Field(default_factory=list)


class Run(RunBase):
    """Run schema when loading from the DB."""

    name: str
    parent_run_id: Optional[UUID]


ChainRun.update_forward_refs()
ToolRun.update_forward_refs()
