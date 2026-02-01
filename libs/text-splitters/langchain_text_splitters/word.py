"""Word text splitter logic."""

from __future__ import annotations

import re
from typing import Any

from langchain_text_splitters.base import TextSplitter


class WordTextSplitter(TextSplitter):
    """Splitting text by words."""

    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        separator: str = " ",
        word_pattern: str | None = None,
        **kwargs: Any,
    ) -> None:
        r"""Create a new WordTextSplitter.

        Args:
            chunk_size: Maximum size of chunks to return (in words).
                Defaults to 4000.
            chunk_overlap: Overlap in words between chunks. Defaults to 200.
            separator: The separator to be used when joining words back
                together. Defaults to " ".
            word_pattern: Regex pattern to identify words. If not provided,
                it defaults to splitting by any whitespace (using re.split).
                To find words using a regex match, provide a pattern that
                captures words (e.g. r"[^\\s]+").
            **kwargs: Additional keyword arguments to customize the splitter.
        """
        if "length_function" not in kwargs:
            kwargs["length_function"] = lambda x: 0 if x == separator else 1

        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self._separator = separator
        self._word_pattern = word_pattern

    def split_text(self, text: str) -> list[str]:
        """Split text into multiple components.

        Args:
            text: The text to split.

        Returns:
            A list of text chunks.
        """
        if not text:
            return []

        # Split text into words
        if self._word_pattern:
            words = re.findall(self._word_pattern, text)
        else:
            # Default behavior: split by whitespace
            words = re.split(r"\s+", text)
            # Remove empty strings from the result of re.split if any
            words = [w for w in words if w]

        # Combine words into chunks
        return self._merge_splits(words, self._separator)
