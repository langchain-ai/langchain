from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from langchain_xfyun.load.serializable import Serializable
from langchain_xfyun.schema.messages import BaseMessage


class PromptValue(Serializable, ABC):
    """Base abstract class for inputs to any language model.

    PromptValues can be converted to both LLM (pure text-generation) inputs and
        ChatModel inputs.
    """

    @property
    def lc_serializable(self) -> bool:
        """
        Return whether or not the class is serializable.
        """
        return True

    @abstractmethod
    def to_string(self) -> str:
        """Return prompt value as string."""

    @abstractmethod
    def to_messages(self) -> List[BaseMessage]:
        """Return prompt as a list of Messages."""
