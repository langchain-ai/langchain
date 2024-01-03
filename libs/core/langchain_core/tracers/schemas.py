"""Schemas for tracers."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from langsmith.schemas import RunBase as BaseRunV2

from langchain_core.pydantic_v1 import Field, root_validator


class Run(BaseRunV2):
    """Run schema for the V2 API in the Tracer."""

    execution_order: int
    child_execution_order: int
    child_runs: List[Run] = Field(default_factory=list)
    tags: Optional[List[str]] = Field(default_factory=list)
    events: List[Dict[str, Any]] = Field(default_factory=list)
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


Run.update_forward_refs()

__all__ = [
    "Run",
]
