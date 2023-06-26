"""Schemas for tracers."""
from __future__ import annotations

import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, root_validator

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


class Run(BaseModel):
    """Run schema for the V2 API in the Tracer."""

    id: UUID
    """The UUID of the run."""
    name: str
    """The name of the run, usually taken from the serialized object's ID."""
    start_time: datetime.datetime
    """The start time of the run."""
    run_type: Union[RunTypeEnum, str]
    """The type of run."""
    inputs: dict
    """The inputs to the run."""
    execution_order: int
    """The order in which this run was executed in a run tree."""
    child_execution_order: int
    """The next execution order of child runs."""
    end_time: Optional[datetime.datetime] = None
    """The end time of the run."""
    extra: Optional[dict] = None
    """Extra information about the run."""
    error: Optional[str] = None
    """The error message of the run, if any."""
    serialized: dict = Field(default_factory=dict)
    """The serialized object that was run."""
    events: Optional[List[Dict]] = None
    """The events that occurred during the run."""
    outputs: Optional[dict] = None
    """The outputs of the run."""
    reference_example_id: Optional[UUID] = None
    """The ID of the reference example that was used to run the run, if this
    run was performed during an evaluation."""
    parent_run_id: Optional[UUID] = None
    """The ID of the parent run if this is not a root."""
    tags: List[str] = Field(default_factory=list)
    """Any tags assigned to the run."""
    session_id: Optional[UUID] = None
    """The Project / Session ID this run belongs to."""
    child_run_ids: Optional[List[UUID]] = None
    """The IDs of the child runs."""
    child_runs: List[Run] = Field(default_factory=list)
    """The child runs. These are used during initial tracing."""
    feedback_stats: Optional[Dict[str, Any]] = None
    """Any feedback statistics for this run."""

    @root_validator(pre=True)
    def assign_name(cls, values: dict) -> dict:
        """Assign name to the run."""
        if values.get("name") is None:
            if "name" in values["serialized"]:
                values["name"] = values["serialized"]["name"]
            elif "id" in values["serialized"]:
                values["name"] = values["serialized"]["id"][-1]
        return values


ChainRun.update_forward_refs()
ToolRun.update_forward_refs()

__all__ = [
    "BaseRun",
    "ChainRun",
    "LLMRun",
    "Run",
    "RunTypeEnum",
    "ToolRun",
    "TracerSession",
    "TracerSessionBase",
    "TracerSessionV1",
    "TracerSessionV1Base",
    "TracerSessionV1Create",
]
