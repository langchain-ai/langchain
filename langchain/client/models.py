from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from langchain.callbacks.tracers.schemas import Run


class Example(BaseModel):
    """Example model."""

    id: UUID
    created_at: datetime
    dataset_id: UUID
    inputs: Dict[str, Any]
    outputs: Optional[Dict[str, Any]] = Field(default=None)
    modified_at: Optional[datetime] = Field(default=None)
    runs: List[Run] = Field(default_factory=list)


class Dataset(BaseModel):
    """Dataset ORM model."""

    id: UUID
    name: str
    description: str
    created_at: datetime
    modified_at: Optional[datetime] = Field(default=None)
    examples: List[Example] = Field(default_factory=list)
