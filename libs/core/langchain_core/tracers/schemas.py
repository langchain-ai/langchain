"""Schemas for tracers."""

from __future__ import annotations

import datetime
import warnings
from typing import Any, Dict, List, Optional, Type
from uuid import UUID

from langsmith.schemas import RunBase as BaseRunV2
from langsmith.schemas import RunTypeEnum as RunTypeEnumDep
from pydantic import PydanticDeprecationWarning
from pydantic.v1 import BaseModel as BaseModelV1
from pydantic.v1 import Field as FieldV1
from pydantic.v1 import root_validator

from langchain_core._api import deprecated


@deprecated("0.1.0", alternative="Use string instead.", removal="1.0")
def RunTypeEnum() -> Type[RunTypeEnumDep]:
    """RunTypeEnum."""
    warnings.warn(
        "RunTypeEnum is deprecated. Please directly use a string instead"
        " (e.g. 'llm', 'chain', 'tool').",
        DeprecationWarning,
        stacklevel=2,
    )
    return RunTypeEnumDep


@deprecated("0.1.0", removal="1.0")
class TracerSessionV1Base(BaseModelV1):
    """Base class for TracerSessionV1."""

    start_time: datetime.datetime = FieldV1(default_factory=datetime.datetime.utcnow)
    name: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


@deprecated("0.1.0", removal="1.0")
class TracerSessionV1Create(TracerSessionV1Base):
    """Create class for TracerSessionV1."""


@deprecated("0.1.0", removal="1.0")
class TracerSessionV1(TracerSessionV1Base):
    """TracerSessionV1 schema."""

    id: int


@deprecated("0.1.0", removal="1.0")
class TracerSessionBase(TracerSessionV1Base):
    """Base class for TracerSession."""

    tenant_id: UUID


@deprecated("0.1.0", removal="1.0")
class TracerSession(TracerSessionBase):
    """TracerSessionV1 schema for the V2 API."""

    id: UUID


@deprecated("0.1.0", alternative="Run", removal="1.0")
class BaseRun(BaseModelV1):
    """Base class for Run."""

    uuid: str
    parent_uuid: Optional[str] = None
    start_time: datetime.datetime = FieldV1(default_factory=datetime.datetime.utcnow)
    end_time: datetime.datetime = FieldV1(default_factory=datetime.datetime.utcnow)
    extra: Optional[Dict[str, Any]] = None
    execution_order: int
    child_execution_order: int
    serialized: Dict[str, Any]
    session_id: int
    error: Optional[str] = None


@deprecated("0.1.0", alternative="Run", removal="1.0")
class LLMRun(BaseRun):
    """Class for LLMRun."""

    prompts: List[str]
    # Temporarily, remove but we will completely remove LLMRun
    # response: Optional[LLMResult] = None


@deprecated("0.1.0", alternative="Run", removal="1.0")
class ChainRun(BaseRun):
    """Class for ChainRun."""

    inputs: Dict[str, Any]
    outputs: Optional[Dict[str, Any]] = None
    child_llm_runs: List[LLMRun] = FieldV1(default_factory=list)
    child_chain_runs: List[ChainRun] = FieldV1(default_factory=list)
    child_tool_runs: List[ToolRun] = FieldV1(default_factory=list)


@deprecated("0.1.0", alternative="Run", removal="1.0")
class ToolRun(BaseRun):
    """Class for ToolRun."""

    tool_input: str
    output: Optional[str] = None
    action: str
    child_llm_runs: List[LLMRun] = FieldV1(default_factory=list)
    child_chain_runs: List[ChainRun] = FieldV1(default_factory=list)
    child_tool_runs: List[ToolRun] = FieldV1(default_factory=list)


# Begin V2 API Schemas


class Run(BaseRunV2):
    """Run schema for the V2 API in the Tracer.

    Parameters:
        child_runs: The child runs.
        tags: The tags. Default is an empty list.
        events: The events. Default is an empty list.
        trace_id: The trace ID. Default is None.
        dotted_order: The dotted order.
    """

    child_runs: List[Run] = FieldV1(default_factory=list)
    tags: Optional[List[str]] = FieldV1(default_factory=list)
    events: List[Dict[str, Any]] = FieldV1(default_factory=list)
    trace_id: Optional[UUID] = None
    dotted_order: Optional[str] = None

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


# TODO: Update once langsmith moves to Pydantic V2 and we can swap Run.model_rebuild
# for Run.update_forward_refs
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=PydanticDeprecationWarning)

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
