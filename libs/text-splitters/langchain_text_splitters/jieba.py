"""Jieba text splitter."""

from __future__ import annotations

from typing import Any

from langchain_text_splitters.base import TextSplitter

try:
    import jieba

    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False


class JiebaTextSplitter(TextSplitter):
    """Splitting text using Jieba package.

    It is good for splitting Chinese text to ensure words are not broken.
    """

    def __init__(
        self,
        separator: str = "",
        **kwargs: Any,
    ) -> None:
        """Initialize the Jieba text splitter."""
        super().__init__(**kwargs)
        self._separator = separator
        if not _HAS_JIEBA:
            msg = """
                Jieba is not installed, please install it with
                `pip install jieba`
                """
            raise ImportError(msg)

    def split_text(self, text: str) -> list[str]:
        """Split incoming text and return chunks."""
        splits = jieba.lcut(text)
        return self._merge_splits(splits, self._separator)
