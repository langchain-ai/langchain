"""Functionality for splitting text."""
from abc import ABC, abstractmethod
from typing import Any, List


class TextSplitter(ABC):
    """Interface for splitting text into chunks."""

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split text into multiple components."""


class BaseChunkTextSplitter(TextSplitter, ABC):
    """Text splitter that first splits into chunks then tries to combine."""

    def __init__(
        self, separator: str = "\n\n", chunk_size: int = 4000, chunk_overlap: int = 200
    ):
        """Initialize with parameters."""
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )
        self._separator = separator
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    @abstractmethod
    def _get_chunk_size(self, text: str) -> int:
        """Return the size of this chunk of text."""

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        # First we naively split the large input into a bunch of smaller ones.
        splits = text.split(self._separator)
        # We now want to combine these smaller pieces into medium size
        # chunks to send to the LLM.
        docs = []
        current_doc: List[str] = []
        total = 0
        for d in splits:
            if total > self._chunk_size:
                docs.append(self._separator.join(current_doc))
                while total > self._chunk_overlap:
                    total -= len(current_doc[0])
                    current_doc = current_doc[1:]
            current_doc.append(d)
            total += len(d)
        docs.append(self._separator.join(current_doc))
        return docs


class CharacterTextSplitter(BaseChunkTextSplitter):
    """Implementation of TextSplitter that uses character length."""

    def _get_chunk_size(self, text: str) -> int:
        return len(text)


class HuggingFaceTokenizerSplitter(BaseChunkTextSplitter):
    """Implementation of TextSplitter that uses HuggingFace tokenizers."""

    def __init__(
        self,
        tokenizer: Any,
        separator: str = "\n\n",
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
    ):
        """Initialize with parameters."""
        try:
            from transformers import PreTrainedTokenizerBase

            if not isinstance(tokenizer, PreTrainedTokenizerBase):
                raise ValueError(
                    "Tokenizer received was not an instance of PreTrainedTokenizerBase"
                )

            self.tokenizer = tokenizer
        except ImportError:
            raise ValueError(
                "Could not import transformers python package. "
                "Please it install it with `pip install transformers`."
            )
        super().__init__(
            separator=separator, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def _get_chunk_size(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
