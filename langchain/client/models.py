from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, root_validator


class Example(BaseModel):
    """Example model."""

    id: str  # TODO: UUID
    created_at: datetime
    dataset_id: str  # TODO: UUID
    inputs: Dict[str, Any]
    outputs: Optional[Dict[str, Any]] = Field(default=None)
    modified_at: Optional[datetime] = Field(default=None)
    llm_runs: List[Dict] = Field(default_factory=list)  # TODO: Type
    chain_runs: List[Dict] = Field(default_factory=list)  # TODO: Type
    runs: List[Dict] = Field(default_factory=list)  # TODO: Type

    @root_validator(pre=True)
    def validate_id(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the ID."""
        # TODO: remove this once we're using UUIDs.
        # We're updating to UUIDs.
        id_ = values.get("id")
        if isinstance(id_, int):
            values["id"] = str(id_)
        return values


class Dataset(BaseModel):
    """Dataset ORM model."""

    id: str
    name: str
    description: str
    created_at: datetime
    modified_at: Optional[datetime] = Field(default=None)
    owner_id: Union[int, str]  # TODO: UUID
    examples: List[Example] = Field(default_factory=list)
