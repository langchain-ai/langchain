"""
It's a sample of refactoring the legacy TextSplitter, to be compatible with lazy, async
and LCEL.
"""
import asyncio
import copy
import logging
import re
from abc import ABC, abstractmethod
from functools import partial
from itertools import cycle
from typing import (
    AbstractSet,
    Any,
    AsyncIterator,
    Callable,
    Collection,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    cast,
)

from langchain.document_transformers.runnable_document_transformer import (
    RunnableGeneratorDocumentTransformer,
    to_sync_iterator,
)
from langchain.pydantic_v1 import root_validator
from langchain.schema import Document
from langchain.text_splitter import TokenTextSplitter

logger = logging.getLogger(__name__)
TS = TypeVar("TS", bound="NewTextSplitter")


class NewTextSplitter(RunnableGeneratorDocumentTransformer, ABC):
    """Interface for splitting text into chunks."""

    chunk_size: int = 4000
    """Maximum size of chunks to return"""
    chunk_overlap: int = 200
    """Overlap in characters between chunks"""
    length_function: Callable[[str], int] = len
    """Function that measures the length of given chunks"""
    keep_separator: bool = False
    """Whether to keep the separator in the chunks"""
    add_start_index: bool = False
    """If `True`, includes chunk's start index in metadata"""
    strip_whitespace: bool = True
    """If `True`, strips whitespace from the start and end of every document"""

    # @model_validator(mode='before')  # pydantic v2
    # @classmethod
    @root_validator(pre=True)  # pydantic v1
    def check_chunk_overlap_and_size(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        chunk_overlap = kwargs["chunk_overlap"]
        chunk_size = kwargs["chunk_size"]
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )
        return kwargs

    # @root_validator(pre=True)
    # def check_chunk_overlap_and_size(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    #     chunk_overlap = kwargs["chunk_overlap"]
    #     chunk_size = kwargs["chunk_size"]
    #     if chunk_overlap > chunk_size:
    #         raise ValueError(
    #             f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
    #             f"({chunk_size}), should be smaller."
    #         )
    #     return kwargs

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split text into multiple components."""

    def lazy_create_documents(
        self, texts: Iterator[str], metadatas: Optional[Iterator[dict]] = None
    ) -> Iterator[Document]:
        """Create documents from an iterator of texts."""
        _metadatas = metadatas or cycle([{}])
        for text, metadata in zip(texts, _metadatas):
            index = -1
            for chunk in self.split_text(text):
                metadata = copy.deepcopy(metadata)
                if self.add_start_index:
                    index = text.find(chunk, index + 1)
                    metadata["start_index"] = index
                new_doc = Document(page_content=chunk, metadata=metadata)
                yield new_doc

    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """Create documents from a list of texts."""
        iter_metadatas = iter(metadatas) if metadatas else None
        return list(
            cast(
                Iterator,
                self.lazy_create_documents(iter(texts), metadatas=iter_metadatas),
            )
        )

    def lazy_split_documents(self, documents: Iterator[Document]) -> Iterator[Document]:
        """Split documents."""
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        for doc in self.lazy_create_documents(iter(texts), metadatas=iter(metadatas)):
            yield doc

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        """Split documents."""
        return list(self.lazy_split_documents(iter(documents)))

    def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
        text = separator.join(docs)
        if self.strip_whitespace:
            text = text.strip()
        if text == "":
            return None
        else:
            return text

    def _merge_splits(self, splits: Iterable[str], separator: str) -> List[str]:
        # We now want to combine these smaller pieces into medium size
        # chunks to send to the LLM.
        separator_len = self.length_function(separator)

        docs = []
        current_doc: List[str] = []
        total = 0
        for d in splits:
            _len = self.length_function(d)
            if (
                total + _len + (separator_len if len(current_doc) > 0 else 0)
                > self.chunk_size
            ):
                if total > self.chunk_size:
                    logger.warning(
                        f"Created a chunk of size {total}, "
                        f"which is longer than the specified {self.chunk_size}"
                    )
                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    # Keep on popping if:
                    # - we have a larger chunk than in the chunk overlap
                    # - or if we still have any chunks and the length is long
                    while total > self.chunk_overlap or (
                        total + _len + (separator_len if len(current_doc) > 0 else 0)
                        > self.chunk_size
                        and total > 0
                    ):
                        total -= self.length_function(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += _len + (separator_len if len(current_doc) > 1 else 0)
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs

    @classmethod
    def from_huggingface_tokenizer(
        cls, tokenizer: Any, **kwargs: Any
    ) -> "NewTextSplitter":
        """Text splitter that uses HuggingFace tokenizer to count length."""
        try:
            from transformers import PreTrainedTokenizerBase

            if not isinstance(tokenizer, PreTrainedTokenizerBase):
                raise ValueError(
                    "Tokenizer received was not an instance of PreTrainedTokenizerBase"
                )

            def _huggingface_tokenizer_length(text: str) -> int:
                return len(tokenizer.encode(text))

        except ImportError:
            raise ValueError(
                "Could not import transformers python package. "
                "Please install it with `pip install transformers`."
            )
        return cls(length_function=_huggingface_tokenizer_length, **kwargs)

    @classmethod
    def from_tiktoken_encoder(
        cls: Type[TS],
        encoding_name: str = "gpt2",
        model_name: Optional[str] = None,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = "all",
        **kwargs: Any,
    ) -> TS:
        """Text splitter that uses tiktoken encoder to count length."""
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "Could not import tiktoken python package. "
                "This is needed in order to calculate max_tokens_for_prompt. "
                "Please install it with `pip install tiktoken`."
            )

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

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Transform sequence of documents by splitting them."""
        return self.split_documents(list(documents))

    async def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Asynchronously transform a sequence of documents by splitting them."""
        return await asyncio.get_running_loop().run_in_executor(
            None, partial(self.transform_documents, **kwargs), documents
        )

    def lazy_transform_documents(
        self, documents: Iterator[Document], **kwargs: Any
    ) -> Iterator[Document]:
        """Transform an iterator of documents.

        Args:
            documents: A sequence of Documents to be transformed.

        Returns:
            An iterator oftransformed Documents.
        """
        return self.lazy_split_documents(documents)

    async def _alazy_transform_documents(  # type: ignore
        self, documents: AsyncIterator[Document], **kwargs: Any
    ) -> AsyncIterator[Document]:
        for doc in self.lazy_split_documents(to_sync_iterator(documents)):
            yield doc


# %% NewCharacterTextSplitter
def _split_text_with_regex(
    text: str, separator: str, keep_separator: bool
) -> List[str]:
    # Now that we have the separator, split the text
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            _splits = re.split(f"({separator})", text)
            splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]
            if len(_splits) % 2 == 0:
                splits += _splits[-1:]
            splits = [_splits[0]] + splits
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


class NewCharacterTextSplitter(NewTextSplitter):
    """Splitting text that looks at characters."""

    separator: str = "\n\n"
    is_separator_regex: bool = False

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        # First we naively split the large input into a bunch of smaller ones.
        separator = (
            self.separator if self.is_separator_regex else re.escape(self.separator)
        )
        splits = _split_text_with_regex(text, separator, self.keep_separator)
        separator = "" if self.keep_separator else self.separator
        return self._merge_splits(splits, separator)


# class NewTokenTextSplitter(NewTextSplitter):
#     """Splitting text to tokens using model tokenizer."""
#
#     encoding_name: str = "gpt2"
#     model_name: Optional[str] = None
#     allowed_special: Union[Literal["all"], AbstractSet[str]] = set()
#     disallowed_special: Union[Literal["all"], Collection[str]] = "all"
#
#     def __init__(
#         self,
#         **kwargs: Any,
#     ) -> None:
#         """Create a new TextSplitter."""
#         super().__init__(**kwargs)
#         try:
#             import tiktoken
#         except ImportError:
#             raise ImportError(
#                 "Could not import tiktoken python package. "
#                 "This is needed in order to for TokenTextSplitter. "
#                 "Please install it with `pip install tiktoken`."
#             )
#
#         if model_name is not None:
#             enc = tiktoken.encoding_for_model(model_name)
#         else:
#             enc = tiktoken.get_encoding(encoding_name)
#         self._tokenizer = enc
#         self._allowed_special = allowed_special
#         self._disallowed_special = disallowed_special
#
#     def split_text(self, text: str) -> List[str]:
#         def _encode(_text: str) -> List[int]:
#             return self._tokenizer.encode(
#                 _text,
#                 allowed_special=self._allowed_special,
#                 disallowed_special=self._disallowed_special,
#             )
#
#         tokenizer = Tokenizer(
#             chunk_overlap=self._chunk_overlap,
#             tokens_per_chunk=self._chunk_size,
#             decode=self._tokenizer.decode,
#             encode=_encode,
#         )
#
#         return split_text_on_tokens(text=text, tokenizer=tokenizer)
