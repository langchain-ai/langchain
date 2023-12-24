from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document


class BaseTextSplitter(ABC):
    """Abstract base class for text splitting."""

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split text into multiple components."""


class BaseTextToDocumentsSplitter(ABC):
    """Abstract base class for splitting text into Documents."""

    @abstractmethod
    def split_text(self, text: str) -> List[Document]:
        """Split text into multiple Documents."""
