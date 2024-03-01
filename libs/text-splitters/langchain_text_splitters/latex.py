from __future__ import annotations

from typing import Any

from langchain_text_splitters.base import Language
from langchain_text_splitters.character import RecursiveCharacterTextSplitter


class LatexTextSplitter(RecursiveCharacterTextSplitter):
    """Attempts to split the text along Latex-formatted layout elements."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a LatexTextSplitter."""
        separators = self.get_separators_for_language(Language.LATEX)
        super().__init__(separators=separators, **kwargs)
