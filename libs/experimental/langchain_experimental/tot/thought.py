from __future__ import annotations

from enum import Enum
from typing import Set

from pydantic import BaseModel, Field


class ThoughtValidity(Enum):
    """Enum for the validity of a thought."""

    VALID_INTERMEDIATE = 0
    VALID_FINAL = 1
    INVALID = 2


class Thought(BaseModel):
    """A thought in the ToT."""

    text: str
    validity: ThoughtValidity
    children: Set[Thought] = Field(default_factory=set)

    def __hash__(self) -> int:
        return id(self)
