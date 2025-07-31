from __future__ import annotations

from typing import Any, Optional, cast

from langchain_text_splitters.base import TextSplitter, Tokenizer, split_text_on_tokens


class TransformersTokenTextSplitter(TextSplitter):
    """Splitting text to tokens using transformers tokenizer.

    This replaces SentenceTransformersTokenTextSplitter by using the transformers
    library directly instead of sentence-transformers, avoiding the heavy
    dependencies of torch and pillow.
    """

    def __init__(
        self,
        chunk_overlap: int = 50,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        tokens_per_chunk: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter using transformers tokenizer.

        Args:
            chunk_overlap: Number of tokens to overlap between chunks.
            model_name: The model name to use for tokenization.
            tokens_per_chunk: Maximum number of tokens per chunk.
            **kwargs: Additional arguments passed to TextSplitter.
        """
        super().__init__(**kwargs, chunk_overlap=chunk_overlap)

        try:
            from transformers import AutoTokenizer  # type: ignore[attr-defined]
        except ImportError:
            msg = (
                "Could not import transformers python package. "
                "This is needed for TransformersTokenTextSplitter. "
                "Please install it with `pip install transformers`."
            )
            raise ImportError(msg)

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, clean_up_tokenization_spaces=False
        )

        # Set a reasonable default if no model_max_length is found
        default_max_length = 512
        self.maximum_tokens_per_chunk = getattr(
            self.tokenizer, "model_max_length", default_max_length
        )

        # Handle cases where model_max_length is very large or None
        if (
            self.maximum_tokens_per_chunk is None
            or self.maximum_tokens_per_chunk > 100000
        ):
            self.maximum_tokens_per_chunk = default_max_length

        self._initialize_chunk_configuration(tokens_per_chunk=tokens_per_chunk)

    def _initialize_chunk_configuration(
        self, *, tokens_per_chunk: Optional[int]
    ) -> None:
        """Initialize the chunk size configuration."""
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

        This method encodes the input text using the transformers tokenizer, then
        strips the start and stop token IDs from the encoded result. It returns the
        processed segments as a list of strings.

        Args:
            text: The input text to be split.

        Returns:
            A list of string components derived from the input text after
            encoding and processing.
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

        This method encodes the input text using the transformers tokenizer and
        calculates the total number of tokens in the encoded result.

        Args:
            text: The input text for which the token count is calculated.

        Returns:
            The number of tokens in the encoded text.
        """
        return len(self._encode(text))

    _max_length_equal_32_bit_integer: int = 2**32

    def _encode(self, text: str) -> list[int]:
        """Encode text using the transformers tokenizer."""
        token_ids_with_start_and_end_token_ids = self.tokenizer.encode(
            text,
            max_length=self._max_length_equal_32_bit_integer,
            truncation=False,
            add_special_tokens=True,
        )
        return cast("list[int]", token_ids_with_start_and_end_token_ids)
