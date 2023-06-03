"""Schemas for tracers."""
from __future__ import annotations

import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, root_validator

from langchain.env import get_runtime_environment
from langchain.schema import LLMResult


class TracerSessionV1Base(BaseModel):
    """Base class for TracerSessionV1."""

    start_time: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    name: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


class TracerSessionV1Create(TracerSessionV1Base):
    """Create class for TracerSessionV1."""


class TracerSessionV1(TracerSessionV1Base):
    """TracerSessionV1 schema."""

    id: int


class TracerSessionBase(TracerSessionV1Base):
    """A creation class for TracerSession."""

    tenant_id: UUID


class TracerSession(TracerSessionBase):
    """TracerSessionV1 schema for the V2 API."""

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


# Begin V2 API Schemas


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
    extra: Optional[Dict[str, Any]] = None
    error: Optional[str]
    execution_order: int
    child_execution_order: Optional[int]
    serialized: dict
    inputs: dict
    outputs: Optional[dict]
    reference_example_id: Optional[UUID]
    run_type: RunTypeEnum
    parent_run_id: Optional[UUID]


class Run(RunBase):
    """Run schema when loading from the DB."""

    name: str
    child_runs: List[Run] = Field(default_factory=list)

    @root_validator(pre=True)
    def assign_name(cls, values: dict) -> dict:
        """Assign name to the run."""
        if "name" not in values:
            values["name"] = values["serialized"]["name"]
        return values


class RunCreate(RunBase):
    name: str
    session_name: Optional[str] = None

    @root_validator(pre=True)
    def add_runtime_env(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Add env info to the run."""
        extra = values.get("extra", {})
        extra["runtime"] = get_runtime_environment()
        values["extra"] = extra
        return values


class RunUpdate(BaseModel):
    end_time: Optional[datetime.datetime]
    error: Optional[str]
    outputs: Optional[dict]
    parent_run_id: Optional[UUID]
    reference_example_id: Optional[UUID]


ChainRun.update_forward_refs()
ToolRun.update_forward_refs()
