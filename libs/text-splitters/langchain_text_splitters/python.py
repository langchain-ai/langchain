from __future__ import annotations

from typing import Any

from langchain_text_splitters.base import Language
from langchain_text_splitters.character import RecursiveCharacterTextSplitter


class PythonCodeTextSplitter(RecursiveCharacterTextSplitter):
    """Attempts to split the text along Python syntax."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a PythonCodeTextSplitter."""
        separators = self.get_separators_for_language(Language.PYTHON)
        super().__init__(separators=separators, **kwargs)
