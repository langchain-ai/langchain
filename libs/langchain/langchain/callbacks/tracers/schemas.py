"""Schemas for tracers."""
from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

from langsmith.schemas import RunBase as BaseRunV2

from langchain.pydantic_v1 import BaseModel, Field, root_validator


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


class Run(BaseRunV2):
    """Run schema for the V2 API in the Tracer."""

    execution_order: int
    child_execution_order: int
    child_runs: List[Run] = Field(default_factory=list)
    tags: Optional[List[str]] = Field(default_factory=list)

    @root_validator(pre=True)
    def assign_name(cls, values: dict) -> dict:
        """Assign name to the run."""
        if values.get("name") is None:
            if "name" in values["serialized"]:
                values["name"] = values["serialized"]["name"]
            elif "id" in values["serialized"]:
                values["name"] = values["serialized"]["id"][-1]
        return values


Run.update_forward_refs()

__all__ = [
    "BaseRun",
    "Run",
]
