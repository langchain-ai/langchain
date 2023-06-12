"""Schemas for tracers."""
from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchainplus_sdk.schemas import Run, RunTypeEnum, RunUpdate
from pydantic import BaseModel, Field

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


def get_run_name(serialized: dict) -> str:
    if "name" in serialized:
        return serialized["name"]
    if "id" in serialized:
        return serialized["id"][-1]
    raise ValueError("Could not find name in serialized run.")


ChainRun.update_forward_refs()
ToolRun.update_forward_refs()

__all__ = [
    "BaseRun",
    "ChainRun",
    "LLMRun",
    "Run",
    "RunTypeEnum",
    "RunUpdate",
    "ToolRun",
    "TracerSession",
    "TracerSessionBase",
    "TracerSessionV1",
    "TracerSessionV1Base",
    "TracerSessionV1Create",
    "get_run_name",
]
