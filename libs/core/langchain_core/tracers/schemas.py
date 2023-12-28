"""Schemas for tracers."""
from __future__ import annotations

import datetime
import warnings
from typing import Any, Dict, List, Optional, Type
from uuid import UUID

from langsmith.schemas import RunBase as BaseRunV2
from langsmith.schemas import RunTypeEnum as RunTypeEnumDep

from langchain_core.outputs import LLMResult
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator


def RunTypeEnum() -> Type[RunTypeEnumDep]:
    """RunTypeEnum."""
    warnings.warn(
        "RunTypeEnum is deprecated. Please directly use a string instead"
        " (e.g. 'llm', 'chain', 'tool').",
        DeprecationWarning,
    )
    return RunTypeEnumDep


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
    """Base class for TracerSession."""

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


class Run(BaseRunV2):
    """Run schema for the V2 API in the Tracer."""

    execution_order: int
    child_execution_order: int
    child_runs: List[Run] = Field(default_factory=list)
    tags: Optional[List[str]] = Field(default_factory=list)
    events: List[Dict[str, Any]] = Field(default_factory=list)

    @root_validator(pre=True)
    def assign_name(cls, values: dict) -> dict:
        """Assign name to the run."""
        if values.get("name") is None:
            if "name" in values["serialized"]:
                values["name"] = values["serialized"]["name"]
            elif "id" in values["serialized"]:
                values["name"] = values["serialized"]["id"][-1]
        if values.get("events") is None:
            values["events"] = []
        return values


ChainRun.update_forward_refs()
ToolRun.update_forward_refs()
Run.update_forward_refs()

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
