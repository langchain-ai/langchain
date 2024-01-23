import pytest

from langchain_core.documents.document_transformers import _LEGACY

# %% -------------------------------------------------------------------------------
# Normally, this class is intended to replace the current implementation of
# TextSplitter. Currently, text_splitter.py is present in langchain, not lanchain-core.
# To demonstrate the use of LCEL for transformers, I therefore need to place this
# example in langchain, not langchain-core.
"""
To demonstrate the possibility of adjusting the existing code to take account of the 
integration of LCEL in the transformers, we propose a new implementation 
of TextSplitter.

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

from langchain.text_splitter import TokenTextSplitter
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import root_validator

from langchain_core.documents.runnable_document_transformer import (
    RunnableGeneratorDocumentTransformer,
    to_sync_iterator,
)

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

# %% -------------------------------------------------------------------------------
# Now, it's time to test this NewCharacterTextSplitter
# This code is a clone of original Test text splitting functionality.
# This is to demonstrate that it is possible to convert legacy implementations
# while maintaining compatibility with current TUs.

CharacterTextSplitter = NewCharacterTextSplitter


def test_character_text_splitter() -> None:
    """Test splitting by character count."""
    text = "foo bar baz 123"
    splitter = CharacterTextSplitter(separator=" ", chunk_size=7, chunk_overlap=3)
    output = splitter.split_text(text)
    expected_output = ["foo bar", "bar baz", "baz 123"]
    assert output == expected_output


def test_character_text_splitter_empty_doc() -> None:
    """Test splitting by character count doesn't create empty documents."""
    text = "foo  bar"
    splitter = CharacterTextSplitter(separator=" ", chunk_size=2, chunk_overlap=0)
    output = splitter.split_text(text)
    expected_output = ["foo", "bar"]
    assert output == expected_output


def test_character_text_splitter_separtor_empty_doc() -> None:
    """Test edge cases are separators."""
    text = "f b"
    splitter = CharacterTextSplitter(separator=" ", chunk_size=2, chunk_overlap=0)
    output = splitter.split_text(text)
    expected_output = ["f", "b"]
    assert output == expected_output


def test_character_text_splitter_long() -> None:
    """Test splitting by character count on long words."""
    text = "foo bar baz a a"
    splitter = CharacterTextSplitter(separator=" ", chunk_size=3, chunk_overlap=1)
    output = splitter.split_text(text)
    expected_output = ["foo", "bar", "baz", "a a"]
    assert output == expected_output


def test_character_text_splitter_short_words_first() -> None:
    """Test splitting by character count when shorter words are first."""
    text = "a a foo bar baz"
    splitter = CharacterTextSplitter(separator=" ", chunk_size=3, chunk_overlap=1)
    output = splitter.split_text(text)
    expected_output = ["a a", "foo", "bar", "baz"]
    assert output == expected_output


def test_character_text_splitter_longer_words() -> None:
    """Test splitting by characters when splits not found easily."""
    text = "foo bar baz 123"
    splitter = CharacterTextSplitter(separator=" ", chunk_size=1, chunk_overlap=1)
    output = splitter.split_text(text)
    expected_output = ["foo", "bar", "baz", "123"]
    assert output == expected_output


@pytest.mark.parametrize(
    "separator, is_separator_regex", [(re.escape("."), True), (".", False)]
)
def test_character_text_splitter_keep_separator_regex(
    separator: str, is_separator_regex: bool
) -> None:
    """Test splitting by characters while keeping the separator
    that is a regex special character.
    """
    text = "foo.bar.baz.123"
    splitter = CharacterTextSplitter(
        separator=separator,
        chunk_size=1,
        chunk_overlap=0,
        keep_separator=True,
        is_separator_regex=is_separator_regex,
    )
    output = splitter.split_text(text)
    expected_output = ["foo", ".bar", ".baz", ".123"]
    assert output == expected_output


@pytest.mark.parametrize(
    "separator, is_separator_regex", [(re.escape("."), True), (".", False)]
)
def test_character_text_splitter_discard_separator_regex(
    separator: str, is_separator_regex: bool
) -> None:
    """Test splitting by characters discarding the separator
    that is a regex special character."""
    text = "foo.bar.baz.123"
    splitter = CharacterTextSplitter(
        separator=separator,
        chunk_size=1,
        chunk_overlap=0,
        keep_separator=False,
        is_separator_regex=is_separator_regex,
    )
    output = splitter.split_text(text)
    expected_output = ["foo", "bar", "baz", "123"]
    assert output == expected_output


def test_character_text_splitting_args() -> None:
    """Test invalid arguments."""
    with pytest.raises(ValueError):
        CharacterTextSplitter(chunk_size=2, chunk_overlap=4)


def test_merge_splits() -> None:
    """Test merging splits with a given separator."""
    splitter = CharacterTextSplitter(separator=" ", chunk_size=9, chunk_overlap=2)
    splits = ["foo", "bar", "baz"]
    expected_output = ["foo bar", "baz"]
    output = splitter._merge_splits(splits, separator=" ")
    assert output == expected_output


def test_create_documents() -> None:
    """Test create documents method."""
    texts = ["foo bar", "baz"]
    splitter = CharacterTextSplitter(separator=" ", chunk_size=3, chunk_overlap=0)
    docs = splitter.create_documents(texts)
    expected_docs = [
        Document(page_content="foo"),
        Document(page_content="bar"),
        Document(page_content="baz"),
    ]
    assert docs == expected_docs


def test_transform_documents() -> None:
    """Test create documents method."""
    texts = ["foo bar", "baz"]
    splitter = CharacterTextSplitter(separator=" ", chunk_size=3, chunk_overlap=0)
    input_docs = [Document(page_content=text) for text in texts]
    docs = splitter.transform_documents(input_docs)
    expected_docs = [
        Document(page_content="foo"),
        Document(page_content="bar"),
        Document(page_content="baz"),
    ]
    assert docs == expected_docs


@pytest.mark.asyncio
async def test_atransform_documents() -> None:
    """Test create documents method."""
    texts = ["foo bar", "baz"]
    splitter = CharacterTextSplitter(separator=" ", chunk_size=3, chunk_overlap=0)
    input_docs = [Document(page_content=text) for text in texts]
    docs = await splitter.atransform_documents(input_docs)
    expected_docs = [
        Document(page_content="foo"),
        Document(page_content="bar"),
        Document(page_content="baz"),
    ]
    assert docs == expected_docs


def test_lazy_transform_documents() -> None:
    """Test create documents method."""
    texts = ["foo bar", "baz"]
    splitter = CharacterTextSplitter(separator=" ", chunk_size=3, chunk_overlap=0)
    input_docs = [Document(page_content=text) for text in texts]
    docs = list(splitter.lazy_transform_documents(iter(input_docs)))
    expected_docs = [
        Document(page_content="foo"),
        Document(page_content="bar"),
        Document(page_content="baz"),
    ]
    assert docs == expected_docs


@pytest.mark.asyncio
async def test_alazy_transform_documents() -> None:
    """Test create documents method."""
    texts = ["foo bar", "baz"]
    splitter = CharacterTextSplitter(separator=" ", chunk_size=3, chunk_overlap=0)
    input_docs = [Document(page_content=text) for text in texts]
    docs = []
    async for doc in splitter.alazy_transform_documents(iter(input_docs)):
        docs.append(doc)
    expected_docs = [
        Document(page_content="foo"),
        Document(page_content="bar"),
        Document(page_content="baz"),
    ]
    assert docs == expected_docs


def test_invoke() -> None:
    """Test create documents method."""
    texts = ["foo bar", "baz"]
    splitter = CharacterTextSplitter(separator=" ", chunk_size=3, chunk_overlap=0)
    input_docs = [Document(page_content=text) for text in texts]
    docs = list(cast(Iterator, splitter.invoke(iter(input_docs))))
    expected_docs = [
        Document(page_content="foo"),
        Document(page_content="bar"),
        Document(page_content="baz"),
    ]
    assert docs == expected_docs


@pytest.mark.asyncio
async def test_ainvoke() -> None:
    """Test create documents method."""
    texts = ["foo bar", "baz"]
    splitter = CharacterTextSplitter(separator=" ", chunk_size=3, chunk_overlap=0)
    input_docs = [Document(page_content=text) for text in texts]
    docs = []
    async for doc in splitter.alazy_transform_documents(iter(input_docs)):
        docs.append(doc)
    expected_docs = [
        Document(page_content="foo"),
        Document(page_content="bar"),
        Document(page_content="baz"),
    ]
    assert docs == expected_docs


def test_create_documents_with_metadata() -> None:
    """Test create documents with metadata method."""
    texts = ["foo bar", "baz"]
    splitter = CharacterTextSplitter(separator=" ", chunk_size=3, chunk_overlap=0)
    docs = splitter.create_documents(texts, [{"source": "1"}, {"source": "2"}])
    expected_docs = [
        Document(page_content="foo", metadata={"source": "1"}),
        Document(page_content="bar", metadata={"source": "1"}),
        Document(page_content="baz", metadata={"source": "2"}),
    ]
    assert docs == expected_docs


def test_create_documents_with_start_index() -> None:
    """Test create documents method."""
    texts = ["foo bar baz 123"]
    splitter = CharacterTextSplitter(
        separator=" ", chunk_size=7, chunk_overlap=3, add_start_index=True
    )
    docs = splitter.create_documents(texts)
    expected_docs = [
        Document(page_content="foo bar", metadata={"start_index": 0}),
        Document(page_content="bar baz", metadata={"start_index": 4}),
        Document(page_content="baz 123", metadata={"start_index": 8}),
    ]
    assert docs == expected_docs


def test_metadata_not_shallow() -> None:
    """Test that metadatas are not shallow."""
    texts = ["foo bar"]
    splitter = CharacterTextSplitter(separator=" ", chunk_size=3, chunk_overlap=0)
    docs = splitter.create_documents(texts, [{"source": "1"}])
    expected_docs = [
        Document(page_content="foo", metadata={"source": "1"}),
        Document(page_content="bar", metadata={"source": "1"}),
    ]
    assert docs == expected_docs
    docs[0].metadata["foo"] = 1
    assert docs[0].metadata == {"source": "1", "foo": 1}
    assert docs[1].metadata == {"source": "1"}


@pytest.mark.skipif(_LEGACY, reason="Test only runnable transformer")
def test_lcel_transform_documents() -> None:
    """Test create documents method."""
    texts = ["foo bar", "baz"]
    runnable = CharacterTextSplitter(
        separator=" ", chunk_size=3, chunk_overlap=0
    ) | CharacterTextSplitter(separator=" ", chunk_size=1, chunk_overlap=0)
    input_docs = [Document(page_content=text) for text in texts]
    docs = list(cast(Iterator[Document], runnable.invoke(input_docs)))
    expected_docs = [
        Document(page_content="foo"),
        Document(page_content="bar"),
        Document(page_content="baz"),
    ]
    assert docs == expected_docs


@pytest.mark.skipif(_LEGACY, reason="Test only runnable transformer")
@pytest.mark.asyncio
async def test_alcel_transform_documents() -> None:
    """Test create documents method."""
    texts = ["foo bar", "baz"]
    runnable = CharacterTextSplitter(
        separator=" ", chunk_size=3, chunk_overlap=0
    ) | CharacterTextSplitter(separator=" ", chunk_size=1, chunk_overlap=0)
    input_docs = [Document(page_content=text) for text in texts]
    docs = [
        doc
        async for doc in cast(
            AsyncIterator[Document], await runnable.ainvoke(input_docs)
        )
    ]
    expected_docs = [
        Document(page_content="foo"),
        Document(page_content="bar"),
        Document(page_content="baz"),
    ]
    assert docs == expected_docs


# def test_iterative_text_splitter_keep_separator() -> None:
#     chunk_size = 5
#     output = __test_iterative_text_splitter(chunk_size=chunk_size, keep_separator=True) # noqa: E501
#
#     assert output == [
#         "....5",
#         "X..3",
#         "Y...4",
#         "X....5",
#         "Y...",
#     ]
#
#
# def test_iterative_text_splitter_discard_separator() -> None:
#     chunk_size = 5
#     output = __test_iterative_text_splitter(chunk_size=chunk_size, keep_separator=False) # noqa: E501
#
#     assert output == [
#         "....5",
#         "..3",
#         "...4",
#         "....5",
#         "...",
#     ]
#
#
# def __test_iterative_text_splitter(chunk_size: int, keep_separator: bool) -> List[str]: # noqa: E501
#     chunk_size += 1 if keep_separator else 0
#
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=0,
#         separators=["X", "Y"],
#         keep_separator=keep_separator,
#     )
#     text = "....5X..3Y...4X....5Y..."
#     output = splitter.split_text(text)
#     for chunk in output:
#         assert len(chunk) <= chunk_size, f"Chunk is larger than {chunk_size}"
#     return output
#
#
# def test_iterative_text_splitter() -> None:
#     """Test iterative text splitter."""
#     text = """Hi.\n\nI'm Harrison.\n\nHow? Are? You?\nOkay then f f f f.
# This is a weird text to write, but gotta test the splittingggg some how.
#
# Bye!\n\n-H."""
#     splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=1)
#     output = splitter.split_text(text)
#     expected_output = [
#         "Hi.",
#         "I'm",
#         "Harrison.",
#         "How? Are?",
#         "You?",
#         "Okay then",
#         "f f f f.",
#         "This is a",
#         "weird",
#         "text to",
#         "write,",
#         "but gotta",
#         "test the",
#         "splitting",
#         "gggg",
#         "some how.",
#         "Bye!",
#         "-H.",
#     ]
#     assert output == expected_output
#


def test_split_documents() -> None:
    """Test split_documents."""
    splitter = CharacterTextSplitter(separator="", chunk_size=1, chunk_overlap=0)
    docs = [
        Document(page_content="foo", metadata={"source": "1"}),
        Document(page_content="bar", metadata={"source": "2"}),
        Document(page_content="baz", metadata={"source": "1"}),
    ]
    expected_output = [
        Document(page_content="f", metadata={"source": "1"}),
        Document(page_content="o", metadata={"source": "1"}),
        Document(page_content="o", metadata={"source": "1"}),
        Document(page_content="b", metadata={"source": "2"}),
        Document(page_content="a", metadata={"source": "2"}),
        Document(page_content="r", metadata={"source": "2"}),
        Document(page_content="b", metadata={"source": "1"}),
        Document(page_content="a", metadata={"source": "1"}),
        Document(page_content="z", metadata={"source": "1"}),
    ]
    assert splitter.split_documents(docs) == expected_output


# def test_python_text_splitter() -> None:
#     splitter = PythonCodeTextSplitter(chunk_size=30, chunk_overlap=0)
#     splits = splitter.split_text(FAKE_PYTHON_TEXT)
#     split_0 = """class Foo:\n\n    def bar():"""
#     split_1 = """def foo():"""
#     split_2 = """def testing_func():"""
#     split_3 = """def bar():"""
#     expected_splits = [split_0, split_1, split_2, split_3]
#     assert splits == expected_splits


CHUNK_SIZE = 16

# def test_python_code_splitter() -> None:
#     splitter = RecursiveCharacterTextSplitter.from_language(
#         Language.PYTHON, chunk_size=CHUNK_SIZE, chunk_overlap=0
#     )
#     code = """
# def hello_world():
#     print("Hello, World!")
#
# # Call the function
# hello_world()
#     """
#     chunks = splitter.split_text(code)
#     assert chunks == [
#         "def",
#         "hello_world():",
#         'print("Hello,',
#         'World!")',
#         "# Call the",
#         "function",
#         "hello_world()",
#     ]
#
#
# def test_golang_code_splitter() -> None:
#     splitter = RecursiveCharacterTextSplitter.from_language(
#         Language.GO, chunk_size=CHUNK_SIZE, chunk_overlap=0
#     )
#     code = """
# package main
#
# import "fmt"
#
# func helloWorld() {
#     fmt.Println("Hello, World!")
# }
#
# func main() {
#     helloWorld()
# }
#     """
#     chunks = splitter.split_text(code)
#     assert chunks == [
#         "package main",
#         'import "fmt"',
#         "func",
#         "helloWorld() {",
#         'fmt.Println("He',
#         "llo,",
#         'World!")',
#         "}",
#         "func main() {",
#         "helloWorld()",
#         "}",
#     ]
#
#
# def test_rst_code_splitter() -> None:
#     splitter = RecursiveCharacterTextSplitter.from_language(
#         Language.RST, chunk_size=CHUNK_SIZE, chunk_overlap=0
#     )
#     code = """
# Sample Document
# ===============
#
# Section
# -------
#
# This is the content of the section.
#
# Lists
# -----
#
# - Item 1
# - Item 2
# - Item 3
#
# Comment
# *******
# Not a comment
#
# .. This is a comment
#     """
#     chunks = splitter.split_text(code)
#     assert chunks == [
#         "Sample Document",
#         "===============",
#         "Section",
#         "-------",
#         "This is the",
#         "content of the",
#         "section.",
#         "Lists",
#         "-----",
#         "- Item 1",
#         "- Item 2",
#         "- Item 3",
#         "Comment",
#         "*******",
#         "Not a comment",
#         ".. This is a",
#         "comment",
#     ]
#     # Special test for special characters
#     code = "harry\n***\nbabylon is"
#     chunks = splitter.split_text(code)
#     assert chunks == ["harry", "***\nbabylon is"]
#
#
# def test_proto_file_splitter() -> None:
#     splitter = RecursiveCharacterTextSplitter.from_language(
#         Language.PROTO, chunk_size=CHUNK_SIZE, chunk_overlap=0
#     )
#     code = """
# syntax = "proto3";
#
# package example;
#
# message Person {
#     string name = 1;
#     int32 age = 2;
#     repeated string hobbies = 3;
# }
#     """
#     chunks = splitter.split_text(code)
#     assert chunks == [
#         "syntax =",
#         '"proto3";',
#         "package",
#         "example;",
#         "message Person",
#         "{",
#         "string name",
#         "= 1;",
#         "int32 age =",
#         "2;",
#         "repeated",
#         "string hobbies",
#         "= 3;",
#         "}",
#     ]
#
#
# def test_javascript_code_splitter() -> None:
#     splitter = RecursiveCharacterTextSplitter.from_language(
#         Language.JS, chunk_size=CHUNK_SIZE, chunk_overlap=0
#     )
#     code = """
# function helloWorld() {
#   console.log("Hello, World!");
# }
#
# // Call the function
# helloWorld();
#     """
#     chunks = splitter.split_text(code)
#     assert chunks == [
#         "function",
#         "helloWorld() {",
#         'console.log("He',
#         "llo,",
#         'World!");',
#         "}",
#         "// Call the",
#         "function",
#         "helloWorld();",
#     ]
#
#
# def test_cobol_code_splitter() -> None:
#     splitter = RecursiveCharacterTextSplitter.from_language(
#         Language.COBOL, chunk_size=CHUNK_SIZE, chunk_overlap=0
#     )
#     code = """
# IDENTIFICATION DIVISION.
# PROGRAM-ID. HelloWorld.
# DATA DIVISION.
# WORKING-STORAGE SECTION.
# 01 GREETING           PIC X(12)   VALUE 'Hello, World!'.
# PROCEDURE DIVISION.
# DISPLAY GREETING.
# STOP RUN.
#     """
#     chunks = splitter.split_text(code)
#     assert chunks == [
#         "IDENTIFICATION",
#         "DIVISION.",
#         "PROGRAM-ID.",
#         "HelloWorld.",
#         "DATA DIVISION.",
#         "WORKING-STORAGE",
#         "SECTION.",
#         "01 GREETING",
#         "PIC X(12)",
#         "VALUE 'Hello,",
#         "World!'.",
#         "PROCEDURE",
#         "DIVISION.",
#         "DISPLAY",
#         "GREETING.",
#         "STOP RUN.",
#     ]
#
#
# def test_typescript_code_splitter() -> None:
#     splitter = RecursiveCharacterTextSplitter.from_language(
#         Language.TS, chunk_size=CHUNK_SIZE, chunk_overlap=0
#     )
#     code = """
# function helloWorld(): void {
#   console.log("Hello, World!");
# }
#
# // Call the function
# helloWorld();
#     """
#     chunks = splitter.split_text(code)
#     assert chunks == [
#         "function",
#         "helloWorld():",
#         "void {",
#         'console.log("He',
#         "llo,",
#         'World!");',
#         "}",
#         "// Call the",
#         "function",
#         "helloWorld();",
#     ]
#
#
# def test_java_code_splitter() -> None:
#     splitter = RecursiveCharacterTextSplitter.from_language(
#         Language.JAVA, chunk_size=CHUNK_SIZE, chunk_overlap=0
#     )
#     code = """
# public class HelloWorld {
#     public static void main(String[] args) {
#         System.out.println("Hello, World!");
#     }
# }
#     """
#     chunks = splitter.split_text(code)
#     assert chunks == [
#         "public class",
#         "HelloWorld {",
#         "public",
#         "static void",
#         "main(String[]",
#         "args) {",
#         "System.out.prin",
#         'tln("Hello,',
#         'World!");',
#         "}\n}",
#     ]
#
#
# def test_kotlin_code_splitter() -> None:
#     splitter = RecursiveCharacterTextSplitter.from_language(
#         Language.KOTLIN, chunk_size=CHUNK_SIZE, chunk_overlap=0
#     )
#     code = """
# class HelloWorld {
#     companion object {
#         @JvmStatic
#         fun main(args: Array<String>) {
#             println("Hello, World!")
#         }
#     }
# }
#     """
#     chunks = splitter.split_text(code)
#     assert chunks == [
#         "class",
#         "HelloWorld {",
#         "companion",
#         "object {",
#         "@JvmStatic",
#         "fun",
#         "main(args:",
#         "Array<String>)",
#         "{",
#         'println("Hello,',
#         'World!")',
#         "}\n    }",
#         "}",
#     ]
#
#
# def test_csharp_code_splitter() -> None:
#     splitter = RecursiveCharacterTextSplitter.from_language(
#         Language.CSHARP, chunk_size=CHUNK_SIZE, chunk_overlap=0
#     )
#     code = """
# using System;
# class Program
# {
#     static void Main()
#     {
#         int age = 30; // Change the age value as needed
#
#         // Categorize the age without any console output
#         if (age < 18)
#         {
#             // Age is under 18
#         }
#         else if (age >= 18 && age < 65)
#         {
#             // Age is an adult
#         }
#         else
#         {
#             // Age is a senior citizen
#         }
#     }
# }
#     """
#
#     chunks = splitter.split_text(code)
#     assert chunks == [
#         "using System;",
#         "class Program\n{",
#         "static void",
#         "Main()",
#         "{",
#         "int age",
#         "= 30; // Change",
#         "the age value",
#         "as needed",
#         "//",
#         "Categorize the",
#         "age without any",
#         "console output",
#         "if (age",
#         "< 18)",
#         "{",
#         "//",
#         "Age is under 18",
#         "}",
#         "else if",
#         "(age >= 18 &&",
#         "age < 65)",
#         "{",
#         "//",
#         "Age is an adult",
#         "}",
#         "else",
#         "{",
#         "//",
#         "Age is a senior",
#         "citizen",
#         "}\n    }",
#         "}",
#     ]
#
#
# def test_cpp_code_splitter() -> None:
#     splitter = RecursiveCharacterTextSplitter.from_language(
#         Language.CPP, chunk_size=CHUNK_SIZE, chunk_overlap=0
#     )
#     code = """
# #include <iostream>
#
# int main() {
#     std::cout << "Hello, World!" << std::endl;
#     return 0;
# }
#     """
#     chunks = splitter.split_text(code)
#     assert chunks == [
#         "#include",
#         "<iostream>",
#         "int main() {",
#         "std::cout",
#         '<< "Hello,',
#         'World!" <<',
#         "std::endl;",
#         "return 0;\n}",
#     ]
#
#
# def test_scala_code_splitter() -> None:
#     splitter = RecursiveCharacterTextSplitter.from_language(
#         Language.SCALA, chunk_size=CHUNK_SIZE, chunk_overlap=0
#     )
#     code = """
# object HelloWorld {
#   def main(args: Array[String]): Unit = {
#     println("Hello, World!")
#   }
# }
#     """
#     chunks = splitter.split_text(code)
#     assert chunks == [
#         "object",
#         "HelloWorld {",
#         "def",
#         "main(args:",
#         "Array[String]):",
#         "Unit = {",
#         'println("Hello,',
#         'World!")',
#         "}\n}",
#     ]
#
#
# def test_ruby_code_splitter() -> None:
#     splitter = RecursiveCharacterTextSplitter.from_language(
#         Language.RUBY, chunk_size=CHUNK_SIZE, chunk_overlap=0
#     )
#     code = """
# def hello_world
#   puts "Hello, World!"
# end
#
# hello_world
#     """
#     chunks = splitter.split_text(code)
#     assert chunks == [
#         "def hello_world",
#         'puts "Hello,',
#         'World!"',
#         "end",
#         "hello_world",
#     ]
#
#
# def test_php_code_splitter() -> None:
#     splitter = RecursiveCharacterTextSplitter.from_language(
#         Language.PHP, chunk_size=CHUNK_SIZE, chunk_overlap=0
#     )
#     code = """
# <?php
# function hello_world() {
#     echo "Hello, World!";
# }
#
# hello_world();
# ?>
#     """
#     chunks = splitter.split_text(code)
#     assert chunks == [
#         "<?php",
#         "function",
#         "hello_world() {",
#         "echo",
#         '"Hello,',
#         'World!";',
#         "}",
#         "hello_world();",
#         "?>",
#     ]
#
#
# def test_swift_code_splitter() -> None:
#     splitter = RecursiveCharacterTextSplitter.from_language(
#         Language.SWIFT, chunk_size=CHUNK_SIZE, chunk_overlap=0
#     )
#     code = """
# func helloWorld() {
#     print("Hello, World!")
# }
#
# helloWorld()
#     """
#     chunks = splitter.split_text(code)
#     assert chunks == [
#         "func",
#         "helloWorld() {",
#         'print("Hello,',
#         'World!")',
#         "}",
#         "helloWorld()",
#     ]
#
#
# def test_rust_code_splitter() -> None:
#     splitter = RecursiveCharacterTextSplitter.from_language(
#         Language.RUST, chunk_size=CHUNK_SIZE, chunk_overlap=0
#     )
#     code = """
# fn main() {
#     println!("Hello, World!");
# }
#     """
#     chunks = splitter.split_text(code)
#     assert chunks == ["fn main() {", 'println!("Hello', ",", 'World!");', "}"]
#
#
# def test_markdown_code_splitter() -> None:
#     splitter = RecursiveCharacterTextSplitter.from_language(
#         Language.MARKDOWN, chunk_size=CHUNK_SIZE, chunk_overlap=0
#     )
#     code = """
# # Sample Document
#
# ## Section
#
# This is the content of the section.
#
# ## Lists
#
# - Item 1
# - Item 2
# - Item 3
#
# ### Horizontal lines
#
# ***********
# ____________
# -------------------
#
# #### Code blocks
# ```
# This is a code block
#
# # sample code
# a = 1
# b = 2
# ```
#     """
#     chunks = splitter.split_text(code)
#     assert chunks == [
#         "# Sample",
#         "Document",
#         "## Section",
#         "This is the",
#         "content of the",
#         "section.",
#         "## Lists",
#         "- Item 1",
#         "- Item 2",
#         "- Item 3",
#         "### Horizontal",
#         "lines",
#         "***********",
#         "____________",
#         "---------------",
#         "----",
#         "#### Code",
#         "blocks",
#         "```",
#         "This is a code",
#         "block",
#         "# sample code",
#         "a = 1\nb = 2",
#         "```",
#     ]
#     # Special test for special characters
#     code = "harry\n***\nbabylon is"
#     chunks = splitter.split_text(code)
#     assert chunks == ["harry", "***\nbabylon is"]
#
#
# def test_latex_code_splitter() -> None:
#     splitter = RecursiveCharacterTextSplitter.from_language(
#         Language.LATEX, chunk_size=CHUNK_SIZE, chunk_overlap=0
#     )
#     code = """
# Hi Harrison!
# \\chapter{1}
# """
#     chunks = splitter.split_text(code)
#     assert chunks == ["Hi Harrison!", "\\chapter{1}"]
#
#
# def test_html_code_splitter() -> None:
#     splitter = RecursiveCharacterTextSplitter.from_language(
#         Language.HTML, chunk_size=60, chunk_overlap=0
#     )
#     code = """
# <h1>Sample Document</h1>
#     <h2>Section</h2>
#         <p id="1234">Reference content.</p>
#
#     <h2>Lists</h2>
#         <ul>
#             <li>Item 1</li>
#             <li>Item 2</li>
#             <li>Item 3</li>
#         </ul>
#
#         <h3>A block</h3>
#             <div class="amazing">
#                 <p>Some text</p>
#                 <p>Some more text</p>
#             </div>
#     """
#     chunks = splitter.split_text(code)
#     assert chunks == [
#         "<h1>Sample Document</h1>\n    <h2>Section</h2>",
#         '<p id="1234">Reference content.</p>',
#         "<h2>Lists</h2>\n        <ul>",
#         "<li>Item 1</li>\n            <li>Item 2</li>",
#         "<li>Item 3</li>\n        </ul>",
#         "<h3>A block</h3>",
#         '<div class="amazing">',
#         "<p>Some text</p>",
#         "<p>Some more text</p>\n            </div>",
#     ]
#
#
# def test_md_header_text_splitter_1() -> None:
#     """Test markdown splitter by header: Case 1."""
#
#     markdown_document = (
#         "# Foo\n\n"
#         "    ## Bar\n\n"
#         "Hi this is Jim\n\n"
#         "Hi this is Joe\n\n"
#         " ## Baz\n\n"
#         " Hi this is Molly"
#     )
#     headers_to_split_on = [
#         ("#", "Header 1"),
#         ("##", "Header 2"),
#     ]
#     markdown_splitter = MarkdownHeaderTextSplitter(
#         headers_to_split_on=headers_to_split_on,
#     )
#     output = markdown_splitter.split_text(markdown_document)
#     expected_output = [
#         Document(
#             page_content="Hi this is Jim  \nHi this is Joe",
#             metadata={"Header 1": "Foo", "Header 2": "Bar"},
#         ),
#         Document(
#             page_content="Hi this is Molly",
#             metadata={"Header 1": "Foo", "Header 2": "Baz"},
#         ),
#     ]
#     assert output == expected_output
#
#
# def test_md_header_text_splitter_2() -> None:
#     """Test markdown splitter by header: Case 2."""
#     markdown_document = (
#         "# Foo\n\n"
#         "    ## Bar\n\n"
#         "Hi this is Jim\n\n"
#         "Hi this is Joe\n\n"
#         " ### Boo \n\n"
#         " Hi this is Lance \n\n"
#         " ## Baz\n\n"
#         " Hi this is Molly"
#     )
#
#     headers_to_split_on = [
#         ("#", "Header 1"),
#         ("##", "Header 2"),
#         ("###", "Header 3"),
#     ]
#     markdown_splitter = MarkdownHeaderTextSplitter(
#         headers_to_split_on=headers_to_split_on,
#     )
#     output = markdown_splitter.split_text(markdown_document)
#     expected_output = [
#         Document(
#             page_content="Hi this is Jim  \nHi this is Joe",
#             metadata={"Header 1": "Foo", "Header 2": "Bar"},
#         ),
#         Document(
#             page_content="Hi this is Lance",
#             metadata={"Header 1": "Foo", "Header 2": "Bar", "Header 3": "Boo"},
#         ),
#         Document(
#             page_content="Hi this is Molly",
#             metadata={"Header 1": "Foo", "Header 2": "Baz"},
#         ),
#     ]
#     assert output == expected_output
#
#
# def test_md_header_text_splitter_3() -> None:
#     """Test markdown splitter by header: Case 3."""
#
#     markdown_document = (
#         "# Foo\n\n"
#         "    ## Bar\n\n"
#         "Hi this is Jim\n\n"
#         "Hi this is Joe\n\n"
#         " ### Boo \n\n"
#         " Hi this is Lance \n\n"
#         " #### Bim \n\n"
#         " Hi this is John \n\n"
#         " ## Baz\n\n"
#         " Hi this is Molly"
#     )
#
#     headers_to_split_on = [
#         ("#", "Header 1"),
#         ("##", "Header 2"),
#         ("###", "Header 3"),
#         ("####", "Header 4"),
#     ]
#
#     markdown_splitter = MarkdownHeaderTextSplitter(
#         headers_to_split_on=headers_to_split_on,
#     )
#     output = markdown_splitter.split_text(markdown_document)
#
#     expected_output = [
#         Document(
#             page_content="Hi this is Jim  \nHi this is Joe",
#             metadata={"Header 1": "Foo", "Header 2": "Bar"},
#         ),
#         Document(
#             page_content="Hi this is Lance",
#             metadata={"Header 1": "Foo", "Header 2": "Bar", "Header 3": "Boo"},
#         ),
#         Document(
#             page_content="Hi this is John",
#             metadata={
#                 "Header 1": "Foo",
#                 "Header 2": "Bar",
#                 "Header 3": "Boo",
#                 "Header 4": "Bim",
#             },
#         ),
#         Document(
#             page_content="Hi this is Molly",
#             metadata={"Header 1": "Foo", "Header 2": "Baz"},
#         ),
#     ]
#
#     assert output == expected_output
#
#
# def test_solidity_code_splitter() -> None:
#     splitter = RecursiveCharacterTextSplitter.from_language(
#         Language.SOL, chunk_size=CHUNK_SIZE, chunk_overlap=0
#     )
#     code = """pragma solidity ^0.8.20;
#   contract HelloWorld {
#     function add(uint a, uint b) pure public returns(uint) {
#       return  a + b;
#     }
#   }
#   """
#     chunks = splitter.split_text(code)
#     assert chunks == [
#         "pragma solidity",
#         "^0.8.20;",
#         "contract",
#         "HelloWorld {",
#         "function",
#         "add(uint a,",
#         "uint b) pure",
#         "public",
#         "returns(uint) {",
#         "return  a",
#         "+ b;",
#         "}\n  }",
#     ]
