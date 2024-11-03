from __future__ import annotations

from typing import Any, List

from langchain_text_splitters.base import TextSplitter


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
        try:
            from konlpy.tag import Kkma
        except ImportError:
            raise ImportError(
                """
                Konlpy is not installed, please install it with 
                `pip install konlpy`
                """
            )
        self.kkma = Kkma()

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        splits = self.kkma.sentences(text)
        return self._merge_splits(splits, self._separator)
