"""Schemas for tracers."""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from langsmith import RunTree
from langsmith.schemas import RunTypeEnum as RunTypeEnumDep
from pydantic import PydanticDeprecationWarning
from pydantic.v1 import BaseModel as BaseModelV1
from pydantic.v1 import Field as FieldV1

from langchain_core._api import deprecated


@deprecated("0.1.0", alternative="Use string instead.", removal="1.0")
def RunTypeEnum() -> type[RunTypeEnumDep]:  # noqa: N802
    """``RunTypeEnum``.

    Returns:
        The ``RunTypeEnum`` class.
    """
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

    start_time: datetime = FieldV1(default_factory=lambda: datetime.now(timezone.utc))
    name: Optional[str] = None
    extra: Optional[dict[str, Any]] = None


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
    start_time: datetime = FieldV1(default_factory=lambda: datetime.now(timezone.utc))
    end_time: datetime = FieldV1(default_factory=lambda: datetime.now(timezone.utc))
    extra: Optional[dict[str, Any]] = None
    execution_order: int
    child_execution_order: int
    serialized: dict[str, Any]
    session_id: int
    error: Optional[str] = None


@deprecated("0.1.0", alternative="Run", removal="1.0")
class LLMRun(BaseRun):
    """Class for LLMRun."""

    prompts: list[str]
    # Temporarily, remove but we will completely remove LLMRun
    # response: Optional[LLMResult] = None


@deprecated("0.1.0", alternative="Run", removal="1.0")
class ChainRun(BaseRun):
    """Class for ChainRun."""

    inputs: dict[str, Any]
    outputs: Optional[dict[str, Any]] = None
    child_llm_runs: list[LLMRun] = FieldV1(default_factory=list)
    child_chain_runs: list[ChainRun] = FieldV1(default_factory=list)
    child_tool_runs: list[ToolRun] = FieldV1(default_factory=list)


@deprecated("0.1.0", alternative="Run", removal="1.0")
class ToolRun(BaseRun):
    """Class for ToolRun."""

    tool_input: str
    output: Optional[str] = None
    action: str
    child_llm_runs: list[LLMRun] = FieldV1(default_factory=list)
    child_chain_runs: list[ChainRun] = FieldV1(default_factory=list)
    child_tool_runs: list[ToolRun] = FieldV1(default_factory=list)


# Begin V2 API Schemas


Run = RunTree  # For backwards compatibility

# TODO: Update once langsmith moves to Pydantic V2 and we can swap Run.model_rebuild
# for Run.update_forward_refs
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=PydanticDeprecationWarning)

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
