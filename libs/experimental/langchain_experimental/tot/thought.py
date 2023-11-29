from __future__ import annotations

from enum import Enum
from typing import Set

from langchain_experimental.pydantic_v1 import BaseModel, Field


class ThoughtValidity(Enum):
    VALID_INTERMEDIATE = 0
    VALID_FINAL = 1
    INVALID = 2


class Thought(BaseModel):
    text: str
    validity: ThoughtValidity
    children: Set[Thought] = Field(default_factory=set)

    def __hash__(self) -> int:
        return id(self)
