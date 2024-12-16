from __future__ import annotations

from typing import Any, List, Optional, cast
from langchain_text_splitters.base import TextSplitter, Tokenizer, split_text_on_tokens

class SentenceTransformersTokenTextSplitter(TextSplitter):
    """Splitting text into tokens using a SentenceTransformer model tokenizer."""

    def __init__(
        self,
        chunk_overlap: int = 50,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        tokens_per_chunk: Optional[int] = None,
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> None:
        """Create a new SentenceTransformersTokenTextSplitter instance.

        Args:
            chunk_overlap (int): The number of overlapping tokens between chunks.
            model_name (str): The name of the SentenceTransformer model.
            tokens_per_chunk (Optional[int]): Maximum tokens per chunk.
            trust_remote_code (bool): Whether to trust remote code when loading the model.
            **kwargs (Any): Additional keyword arguments.
        """

        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "Could not import the transformers package. "
                "Install it with `pip install transformers`."
            )
            
        super().__init__(**kwargs, chunk_overlap=chunk_overlap)

        self.model_name = model_name

        self.tokenizer = tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=trust_remote_code)
        self._initialize_chunk_configuration(tokens_per_chunk=tokens_per_chunk)

    def _initialize_chunk_configuration(self, *, tokens_per_chunk: Optional[int]) -> None:
        """Initialize chunk-related configuration based on the model's limits."""
        self.maximum_tokens_per_chunk = cast(int, self.tokenizer.model_max_length)

        if tokens_per_chunk is None:
            self.tokens_per_chunk = self.maximum_tokens_per_chunk
        else:
            self.tokens_per_chunk = tokens_per_chunk

        if self.tokens_per_chunk > self.maximum_tokens_per_chunk:
            raise ValueError(
                f"The token limit of the model '{self.model_name}' is: {self.maximum_tokens_per_chunk}. "
                f"Argument tokens_per_chunk={self.tokens_per_chunk} exceeds this limit."
            )

    def split_text(self, text: str) -> List[str]:
        """Split the input text into smaller chunks based on the tokenizer configuration."""

        def encode_strip_start_and_stop_token_ids(text: str) -> List[int]:
            """Encode text and strip start and stop token IDs."""
            return self._encode(text)[1:-1]

        tokenizer = Tokenizer(
            chunk_overlap=self._chunk_overlap,
            tokens_per_chunk=self.tokens_per_chunk,
            decode=self.tokenizer.decode,
            encode=encode_strip_start_and_stop_token_ids,
        )

        return split_text_on_tokens(text=text, tokenizer=tokenizer)

    def count_tokens(self, *, text: str) -> int:
        """Count the number of tokens in the input text."""
        return len(self._encode(text))

    _max_length_equal_32_bit_integer: int = 2**32

    def _encode(self, text: str) -> List[int]:
        """Encode text into a list of token IDs."""
        return self.tokenizer.encode(
            text,
            max_length=self._max_length_equal_32_bit_integer,
            truncation="do_not_truncate",
        )
