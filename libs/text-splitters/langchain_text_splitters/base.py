"""Text splitter base interface."""

from __future__ import annotations

import copy
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeVar,
)

from langchain_core.documents import BaseDocumentTransformer, Document
from typing_extensions import Self, override

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Sequence
    from collections.abc import Set as AbstractSet


try:
    import tiktoken

    _HAS_TIKTOKEN = True
except ImportError:
    _HAS_TIKTOKEN = False

try:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

logger = logging.getLogger(__name__)

TS = TypeVar("TS", bound="TextSplitter")


class TextSplitter(BaseDocumentTransformer, ABC):
    """Interface for splitting text into chunks."""

    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
        keep_separator: bool | Literal["start", "end"] = False,  # noqa: FBT001,FBT002
        add_start_index: bool = False,  # noqa: FBT001,FBT002
        strip_whitespace: bool = True,  # noqa: FBT001,FBT002
        metadata_hydrator: Callable[[Document, int], dict[str, Any]] | None = None,
    ) -> None:
        """Create a new `TextSplitter`.

        Args:
            chunk_size: Maximum size of chunks to return
            chunk_overlap: Overlap in characters between chunks
            length_function: Function that measures the length of given chunks
            keep_separator: Whether to keep the separator and where to place it
                in each corresponding chunk `(True='start')`
            add_start_index: If `True`, includes chunk's start index in metadata
            strip_whitespace: If `True`, strips whitespace from the start and end of
                every document
            metadata_hydrator: Optional function to enrich chunk metadata.
                Takes (Document, chunk_index) and returns a dict of metadata to add.

        Raises:
            ValueError: If `chunk_size` is less than or equal to 0
            ValueError: If `chunk_overlap` is less than 0
            ValueError: If `chunk_overlap` is greater than `chunk_size`
        """
        if chunk_size <= 0:
            msg = f"chunk_size must be > 0, got {chunk_size}"
            raise ValueError(msg)
        if chunk_overlap < 0:
            msg = f"chunk_overlap must be >= 0, got {chunk_overlap}"
            raise ValueError(msg)
        if chunk_overlap > chunk_size:
            msg = (
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )
            raise ValueError(msg)
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._keep_separator = keep_separator
        self._add_start_index = add_start_index
        self._strip_whitespace = strip_whitespace
        self._metadata_hydrator = metadata_hydrator

    @abstractmethod
    def split_text(self, text: str) -> list[str]:
        """Split text into multiple components.

        Args:
            text: The text to split.

        Returns:
            A list of text chunks.
        """

    def create_documents(
        self, texts: list[str], metadatas: list[dict[Any, Any]] | None = None
    ) -> list[Document]:
        """Create a list of `Document` objects from a list of texts.

        Args:
            texts: A list of texts to be split and converted into documents.
            metadatas: Optional list of metadata to associate with each document.

        Returns:
            A list of `Document` objects.
        """
        metadatas_ = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            index = 0
            previous_chunk_len = 0
            for chunk_index, chunk in enumerate(self.split_text(text)):
                metadata = copy.deepcopy(metadatas_[i])
                if self._add_start_index:
                    offset = index + previous_chunk_len - self._chunk_overlap
                    index = text.find(chunk, max(0, offset))
                    metadata["start_index"] = index
                    previous_chunk_len = len(chunk)
                new_doc = Document(page_content=chunk, metadata=metadata)
                if self._metadata_hydrator is not None:
                    hydrated_metadata = self._metadata_hydrator(new_doc, chunk_index)
                    if hydrated_metadata:
                        new_doc.metadata.update(hydrated_metadata)
                documents.append(new_doc)
        return documents

    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        """Split documents.

        Args:
            documents: The documents to split.

        Returns:
            A list of split documents.
        """
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        return self.create_documents(texts, metadatas=metadatas)

    def _join_docs(self, docs: list[str], separator: str) -> str | None:
        text = separator.join(docs)
        if self._strip_whitespace:
            text = text.strip()
        return text or None

    def _merge_splits(self, splits: Iterable[str], separator: str) -> list[str]:
        # We now want to combine these smaller pieces into medium size
        # chunks to send to the LLM.
        separator_len = self._length_function(separator)

        docs = []
        current_doc: list[str] = []
        total = 0
        for d in splits:
            len_ = self._length_function(d)
            if (
                total + len_ + (separator_len if len(current_doc) > 0 else 0)
                > self._chunk_size
            ):
                if total > self._chunk_size:
                    logger.warning(
                        "Created a chunk of size %d, which is longer than the "
                        "specified %d",
                        total,
                        self._chunk_size,
                    )
                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    # Keep on popping if:
                    # - we have a larger chunk than in the chunk overlap
                    # - or if we still have any chunks and the length is long
                    while total > self._chunk_overlap or (
                        total + len_ + (separator_len if len(current_doc) > 0 else 0)
                        > self._chunk_size
                        and total > 0
                    ):
                        total -= self._length_function(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += len_ + (separator_len if len(current_doc) > 1 else 0)
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs

    @classmethod
    def from_huggingface_tokenizer(
        cls, tokenizer: PreTrainedTokenizerBase, **kwargs: Any
    ) -> TextSplitter:
        """Text splitter that uses Hugging Face tokenizer to count length.

        Args:
            tokenizer: The Hugging Face tokenizer to use.

        Returns:
            An instance of `TextSplitter` using the Hugging Face tokenizer for length
                calculation.
        """
        if not _HAS_TRANSFORMERS:
            msg = (
                "Could not import transformers python package. "
                "Please install it with `pip install transformers`."
            )
            raise ValueError(msg)

        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            msg = "Tokenizer received was not an instance of PreTrainedTokenizerBase"  # type: ignore[unreachable]
            raise ValueError(msg)  # noqa: TRY004

        def _huggingface_tokenizer_length(text: str) -> int:
            return len(tokenizer.tokenize(text))

        return cls(length_function=_huggingface_tokenizer_length, **kwargs)

    @classmethod
    def from_tiktoken_encoder(
        cls,
        encoding_name: str = "gpt2",
        model_name: str | None = None,
        allowed_special: Literal["all"] | AbstractSet[str] = set(),
        disallowed_special: Literal["all"] | Collection[str] = "all",
        **kwargs: Any,
    ) -> Self:
        """Text splitter that uses `tiktoken` encoder to count length.

        Args:
            encoding_name: The name of the tiktoken encoding to use.
            model_name: The name of the model to use.

                If provided, this will override the `encoding_name`.
            allowed_special: Special tokens that are allowed during encoding.
            disallowed_special: Special tokens that are disallowed during encoding.

        Returns:
            An instance of `TextSplitter` using tiktoken for length calculation.

        Raises:
            ImportError: If the tiktoken package is not installed.
        """
        if not _HAS_TIKTOKEN:
            msg = (
                "Could not import tiktoken python package. "
                "This is needed in order to calculate max_tokens_for_prompt. "
                "Please install it with `pip install tiktoken`."
            )
            raise ImportError(msg)

        if model_name is not None:
            enc = tiktoken.encoding_for_model(model_name)
        else:
            enc = tiktoken.get_encoding(encoding_name)

        def _tiktoken_encoder(text: str) -> int:
            return len(
                enc.encode(
                    text,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )

        if issubclass(cls, TokenTextSplitter):
            extra_kwargs = {
                "encoding_name": encoding_name,
                "model_name": model_name,
                "allowed_special": allowed_special,
                "disallowed_special": disallowed_special,
            }
            kwargs = {**kwargs, **extra_kwargs}

        return cls(length_function=_tiktoken_encoder, **kwargs)

    @override
    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Transform sequence of documents by splitting them.

        Args:
            documents: The sequence of documents to split.

        Returns:
            A list of split documents.
        """
        return self.split_documents(list(documents))


class TokenTextSplitter(TextSplitter):
    """Splitting text to tokens using model tokenizer."""

    def __init__(
        self,
        encoding_name: str = "gpt2",
        model_name: str | None = None,
        allowed_special: Literal["all"] | AbstractSet[str] = set(),
        disallowed_special: Literal["all"] | Collection[str] = "all",
        **kwargs: Any,
    ) -> None:
        """Create a new `TextSplitter`.

        Args:
            encoding_name: The name of the tiktoken encoding to use.
            model_name: The name of the model to use.

                If provided, this will override the `encoding_name`.
            allowed_special: Special tokens that are allowed during encoding.
            disallowed_special: Special tokens that are disallowed during encoding.

        Raises:
            ImportError: If the tiktoken package is not installed.
        """
        super().__init__(**kwargs)
        if not _HAS_TIKTOKEN:
            msg = (
                "Could not import tiktoken python package. "
                "This is needed in order to for TokenTextSplitter. "
                "Please install it with `pip install tiktoken`."
            )
            raise ImportError(msg)

        if model_name is not None:
            enc = tiktoken.encoding_for_model(model_name)
        else:
            enc = tiktoken.get_encoding(encoding_name)
        self._tokenizer = enc
        self._allowed_special = allowed_special
        self._disallowed_special = disallowed_special

    def split_text(self, text: str) -> list[str]:
        """Splits the input text into smaller chunks based on tokenization.

        This method uses a custom tokenizer configuration to encode the input text
        into tokens, processes the tokens in chunks of a specified size with overlap,
        and decodes them back into text chunks. The splitting is performed using the
        `split_text_on_tokens` function.

        Args:
            text: The input text to be split into smaller chunks.

        Returns:
            A list of text chunks, where each chunk is derived from a portion
                of the input text based on the tokenization and chunking rules.
        """

        def _encode(_text: str) -> list[int]:
            return self._tokenizer.encode(
                _text,
                allowed_special=self._allowed_special,
                disallowed_special=self._disallowed_special,
            )

        tokenizer = Tokenizer(
            chunk_overlap=self._chunk_overlap,
            tokens_per_chunk=self._chunk_size,
            decode=self._tokenizer.decode,
            encode=_encode,
        )

        return split_text_on_tokens(text=text, tokenizer=tokenizer)


class Language(str, Enum):
    """Enum of the programming languages."""

    CPP = "cpp"
    GO = "go"
    JAVA = "java"
    KOTLIN = "kotlin"
    JS = "js"
    TS = "ts"
    PHP = "php"
    PROTO = "proto"
    PYTHON = "python"
    R = "r"
    RST = "rst"
    RUBY = "ruby"
    RUST = "rust"
    SCALA = "scala"
    SWIFT = "swift"
    MARKDOWN = "markdown"
    LATEX = "latex"
    HTML = "html"
    SOL = "sol"
    CSHARP = "csharp"
    COBOL = "cobol"
    C = "c"
    LUA = "lua"
    PERL = "perl"
    HASKELL = "haskell"
    ELIXIR = "elixir"
    POWERSHELL = "powershell"
    VISUALBASIC6 = "visualbasic6"


@dataclass(frozen=True)
class Tokenizer:
    """Tokenizer data class."""

    chunk_overlap: int
    """Overlap in tokens between chunks"""

    tokens_per_chunk: int
    """Maximum number of tokens per chunk"""

    decode: Callable[[list[int]], str]
    """ Function to decode a list of token IDs to a string"""

    encode: Callable[[str], list[int]]
    """ Function to encode a string to a list of token IDs"""


def split_text_on_tokens(*, text: str, tokenizer: Tokenizer) -> list[str]:
    """Split incoming text and return chunks using tokenizer.

    Args:
        text: The input text to be split.
        tokenizer: The tokenizer to use for splitting.

    Returns:
        A list of text chunks.
    """
    splits: list[str] = []
    input_ids = tokenizer.encode(text)
    start_idx = 0
    if tokenizer.tokens_per_chunk <= tokenizer.chunk_overlap:
        msg = "tokens_per_chunk must be greater than chunk_overlap"
        raise ValueError(msg)

    while start_idx < len(input_ids):
        cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]
        if not chunk_ids:
            break
        decoded = tokenizer.decode(chunk_ids)
        if decoded:
            splits.append(decoded)
        if cur_idx == len(input_ids):
            break
        start_idx += tokenizer.tokens_per_chunk - tokenizer.chunk_overlap
    return splits
