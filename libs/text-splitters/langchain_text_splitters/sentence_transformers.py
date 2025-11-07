"""Sentence transformers text splitter."""

from __future__ import annotations

from typing import Any, cast

from langchain_text_splitters.base import TextSplitter, Tokenizer, split_text_on_tokens

try:
    # Type ignores needed as long as sentence-transformers doesn't support Python 3.14.
    from sentence_transformers import (  # type: ignore[import-not-found, unused-ignore]
        SentenceTransformer,
    )

    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False


class SentenceTransformersTokenTextSplitter(TextSplitter):
    """Splitting text to tokens using sentence model tokenizer."""

    def __init__(
        self,
        chunk_overlap: int = 50,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        tokens_per_chunk: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(**kwargs, chunk_overlap=chunk_overlap)

        if not _HAS_SENTENCE_TRANSFORMERS:
            msg = (
                "Could not import sentence_transformers python package. "
                "This is needed in order to use SentenceTransformersTokenTextSplitter. "
                "Please install it with `pip install sentence-transformers`."
            )
            raise ImportError(msg)

        self.model_name = model_name
        self._model = SentenceTransformer(self.model_name)
        self.tokenizer = self._model.tokenizer
        self._initialize_chunk_configuration(tokens_per_chunk=tokens_per_chunk)

    def _initialize_chunk_configuration(self, *, tokens_per_chunk: int | None) -> None:
        self.maximum_tokens_per_chunk = self._model.max_seq_length

        if tokens_per_chunk is None:
            self.tokens_per_chunk = self.maximum_tokens_per_chunk
        else:
            self.tokens_per_chunk = tokens_per_chunk

        if self.tokens_per_chunk > self.maximum_tokens_per_chunk:
            msg = (
                f"The token limit of the models '{self.model_name}'"
                f" is: {self.maximum_tokens_per_chunk}."
                f" Argument tokens_per_chunk={self.tokens_per_chunk}"
                f" > maximum token limit."
            )
            raise ValueError(msg)

    def split_text(self, text: str) -> list[str]:
        """Splits the input text into smaller components by splitting text on tokens.

        This method encodes the input text using a private `_encode` method, then
        strips the start and stop token IDs from the encoded result. It returns the
        processed segments as a list of strings.

        Args:
            text: The input text to be split.

        Returns:
            A list of string components derived from the input text after encoding and
            processing.
        """

        def encode_strip_start_and_stop_token_ids(text: str) -> list[int]:
            return self._encode(text)[1:-1]

        tokenizer = Tokenizer(
            chunk_overlap=self._chunk_overlap,
            tokens_per_chunk=self.tokens_per_chunk,
            decode=self.tokenizer.decode,
            encode=encode_strip_start_and_stop_token_ids,
        )

        return split_text_on_tokens(text=text, tokenizer=tokenizer)

    def count_tokens(self, *, text: str) -> int:
        """Counts the number of tokens in the given text.

        This method encodes the input text using a private `_encode` method and
        calculates the total number of tokens in the encoded result.

        Args:
            text: The input text for which the token count is calculated.

        Returns:
            int: The number of tokens in the encoded text.
        """
        return len(self._encode(text))

    _max_length_equal_32_bit_integer: int = 2**32

    def _encode(self, text: str) -> list[int]:
        token_ids_with_start_and_end_token_ids = self.tokenizer.encode(
            text,
            max_length=self._max_length_equal_32_bit_integer,
            truncation="do_not_truncate",
        )
        return cast("list[int]", token_ids_with_start_and_end_token_ids)
