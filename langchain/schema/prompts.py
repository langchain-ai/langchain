from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from pydantic import BaseModel

from langchain.schema.messages import BaseMessage


class PromptValue(BaseModel, ABC):
    @abstractmethod
    def to_string(self) -> str:
        """Return prompt as string."""

    @abstractmethod
    def to_messages(self) -> List[BaseMessage]:
        """Return prompt as messages."""
