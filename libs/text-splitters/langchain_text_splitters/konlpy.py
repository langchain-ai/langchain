"""Konlpy text splitter."""

from __future__ import annotations

from typing import Any

from langchain_text_splitters.base import TextSplitter

try:
    import konlpy

    _HAS_KONLPY = True
except ImportError:
    _HAS_KONLPY = False


class KonlpyTextSplitter(TextSplitter):
    """Splitting text using Konlpy package.

    It is good for splitting Korean text.
    """

    def __init__(
        self,
        separator: str = "\n\n",
        **kwargs: Any,
    ) -> None:
        """Initialize the Konlpy text splitter."""
        super().__init__(**kwargs)
        self._separator = separator
        if not _HAS_KONLPY:
            msg = """
                Konlpy is not installed, please install it with
                `pip install konlpy`
                """
            raise ImportError(msg)
        self.kkma = konlpy.tag.Kkma()

    def split_text(self, text: str) -> list[str]:
        """Split incoming text and return chunks."""
        splits = self.kkma.sentences(text)
        return self._merge_splits(splits, self._separator)
