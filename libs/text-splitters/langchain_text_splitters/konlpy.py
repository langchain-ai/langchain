"""Konlpy text splitter."""

from __future__ import annotations

from typing import Any

from typing_extensions import override

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
        """Initialize the Konlpy text splitter.

        Args:
            separator: The separator to use when combining splits.

        Raises:
            ImportError: If Konlpy is not installed.
        """
        super().__init__(**kwargs)
        self._separator = separator
        try:
            import konlpy  # noqa: PLC0415
        except ImportError:
            msg = (
                "Konlpy is not installed, please install it with `pip install konlpy`."
            )
            raise ImportError(msg) from None
        self.kkma = konlpy.tag.Kkma()

    @override
    def split_text(self, text: str) -> list[str]:
        splits = self.kkma.sentences(text)
        return self._merge_splits(splits, self._separator)
