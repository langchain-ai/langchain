"""**Text Splitters** are classes for splitting text.


**Class hierarchy:**

.. code-block::

    BaseDocumentTransformer --> TextSplitter --> <name>TextSplitter  # Example: CharacterTextSplitter
                                                 RecursiveCharacterTextSplitter -->  <name>TextSplitter

Note: **MarkdownHeaderTextSplitter** and **HTMLHeaderTextSplitter do not derive from TextSplitter.


**Main helpers:**

.. code-block::

    Document, Tokenizer, Language, LineType, HeaderType

"""  # noqa: E501

from __future__ import annotations

import asyncio
import copy
import logging
import pathlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import partial
from io import BytesIO, StringIO
from typing import (
    AbstractSet,
    Any,
    Callable,
    Collection,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    Union,
    cast,
)

import requests
from langchain_core.documents import BaseDocumentTransformer

from langchain.docstore.document import Document
from langchain.pydantic_v1 import BaseModel

logger = logging.getLogger(__name__)

TS = TypeVar("TS", bound="TextSplitter")


class SplittedText(BaseModel):
    text: str
    is_separator: bool = False

    def __len__(self) -> int:
        return len(self.text)

    def __add__(self, other: SplittedText) -> SplittedText:
        return SplittedText(text=self.text + other.text)


def _make_spacy_pipeline_for_splitting(pipeline: str) -> Any:  # avoid importing spacy
    try:
        import spacy
    except ImportError:
        raise ImportError(
            "Spacy is not installed, please install it with `pip install spacy`."
        )
    if pipeline == "sentencizer":
        from spacy.lang.en import English

        sentencizer = English()
        sentencizer.add_pipe("sentencizer")
    else:
        sentencizer = spacy.load(pipeline, exclude=["ner", "tagger"])
    return sentencizer


def _split_text_with_regex(text: str, separator: str) -> List[SplittedText]:
    """Split text using regex and clean empty strings."""
    splits = re.split(separator, text)
    splits = list(filter(lambda s: s != "", splits))

    # force is_separator=False when the separator is empty
    return [
        SplittedText(
            text=split,
            is_separator=bool(re.match(separator, split)) and bool(separator),
        )
        for split in splits
    ]


class TextSplitter(BaseDocumentTransformer, ABC):
    """Interface for splitting text into chunks."""

    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
        keep_separator: bool = False,
        add_start_index: bool = False,
        strip_whitespace: bool = True,
    ) -> None:
        """Create a new TextSplitter.

        Args:
            chunk_size: Maximum size of chunks to return
            chunk_overlap: Overlap in characters between chunks
            length_function: Function that measures the length of given chunks
            keep_separator: Whether to keep the separator in the chunks
            add_start_index: If `True`, includes chunk's start index in metadata
            strip_whitespace: If `True`, strips whitespace from the start and end of
                              every document
        """
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._keep_separator = keep_separator
        self._add_start_index = add_start_index
        self._strip_whitespace = strip_whitespace

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split text into multiple components."""

    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """Create documents from a list of texts."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            index = -1
            for chunk in self.split_text(text):
                metadata = copy.deepcopy(_metadatas[i])
                if self._add_start_index:
                    index = text.find(chunk, index + 1)
                    metadata["start_index"] = index
                new_doc = Document(page_content=chunk, metadata=metadata)
                documents.append(new_doc)
        return documents

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        """Split documents."""
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        return self.create_documents(texts, metadatas=metadatas)

    def _join_splits(self, splits: List[SplittedText]) -> str:
        splits_str = [split.text for split in splits]
        text = "".join(splits_str)
        if self._strip_whitespace:
            text = text.strip()
        return text

    def _merge_splits(self, splits: List[SplittedText]) -> List[str]:
        """Merge splits into chunks.

        Args:
            splits: List of splits to merge

        Returns:
            List of chunks
        """
        if self._keep_separator:
            # flag separators as ordinary text
            splits = [SplittedText(text=split.text) for split in splits]
        else:
            # remove separators from the beginning
            while splits and splits[0].is_separator:
                splits.pop(0)
            # remove separators from the end
            while splits and splits[-1].is_separator:
                splits.pop(-1)

        if not splits:
            return []

        docs = []
        left_split_idx = 0
        right_split_idx = 0
        next_split_idx = 0
        current_doc_length = len(splits[0])
        next_split_length = 0

        def append_doc() -> None:
            nonlocal left_split_idx, right_split_idx, current_doc_length

            if left_split_idx > right_split_idx:
                return

            while (
                right_split_idx > left_split_idx
                and current_doc_length > self._chunk_size
            ) or splits[left_split_idx].is_separator:
                current_doc_length -= len(splits[left_split_idx])
                left_split_idx += 1

            if current_doc_length > self._chunk_size:
                logger.warning(
                    f"Created a chunk of size {current_doc_length}, which is longer "
                    f"than the specified {self._chunk_size}"
                )

            # create a new chunk
            doc = self._join_splits(splits[left_split_idx : right_split_idx + 1])
            if doc:
                docs.append(doc)

        while next_split_idx < len(splits) - 1:
            # make next_split_idx point to the first non-separator split
            next_split_idx += 1
            next_split_length += len(splits[next_split_idx])
            while next_split_idx < len(splits) and splits[next_split_idx].is_separator:
                next_split_idx += 1
                next_split_length += len(splits[next_split_idx])

            # append doc
            if current_doc_length + next_split_length > self._chunk_size:
                append_doc()

                # reduce doc length to self._chunk_overlap by removing splits from the
                # left
                while (
                    right_split_idx >= left_split_idx
                    and current_doc_length > self._chunk_overlap
                ):
                    current_doc_length -= len(splits[left_split_idx])
                    left_split_idx += 1

            right_split_idx = next_split_idx
            current_doc_length += next_split_length
            next_split_length = 0

        append_doc()

        return docs

    @classmethod
    def from_huggingface_tokenizer(cls, tokenizer: Any, **kwargs: Any) -> TextSplitter:
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
        allowed_special: Optional[Union[Literal["all"], AbstractSet[str]]] = None,
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

        allowed_special = allowed_special or set()

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


class CharacterTextSplitter(TextSplitter):
    """Splitting text that looks at characters."""

    def __init__(
        self, separator: str = "\n\n", is_separator_regex: bool = False, **kwargs: Any
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(**kwargs)
        self._separator = separator
        self._is_separator_regex = is_separator_regex

        # If the separator is a regex, we don't need to escape it.
        if not self._is_separator_regex:
            self._separator = re.escape(self._separator)
        # Capture separators.
        if self._separator:
            self._separator = f"({self._separator})"

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        splits = _split_text_with_regex(text, self._separator)
        return self._merge_splits(splits)


class LineType(TypedDict):
    """Line type as typed dict."""

    metadata: Dict[str, str]
    content: str


class HeaderType(TypedDict):
    """Header type as typed dict."""

    level: int
    name: str
    data: str


class MarkdownHeaderTextSplitter:
    """Splitting markdown files based on specified headers."""

    def __init__(
        self, headers_to_split_on: List[Tuple[str, str]], return_each_line: bool = False
    ):
        """Create a new MarkdownHeaderTextSplitter.

        Args:
            headers_to_split_on: Headers we want to track
            return_each_line: Return each line w/ associated headers
        """
        # Output line-by-line or aggregated into chunks w/ common headers
        self.return_each_line = return_each_line
        # Given the headers we want to split on,
        # (e.g., "#, ##, etc") order by length
        self.headers_to_split_on = sorted(
            headers_to_split_on, key=lambda split: len(split[0]), reverse=True
        )

    def aggregate_lines_to_chunks(self, lines: List[LineType]) -> List[Document]:
        """Combine lines with common metadata into chunks
        Args:
            lines: Line of text / associated header metadata
        """
        aggregated_chunks: List[LineType] = []

        for line in lines:
            if (
                aggregated_chunks
                and aggregated_chunks[-1]["metadata"] == line["metadata"]
            ):
                # If the last line in the aggregated list
                # has the same metadata as the current line,
                # append the current content to the last lines's content
                aggregated_chunks[-1]["content"] += "  \n" + line["content"]
            else:
                # Otherwise, append the current line to the aggregated list
                aggregated_chunks.append(line)

        return [
            Document(page_content=chunk["content"], metadata=chunk["metadata"])
            for chunk in aggregated_chunks
        ]

    def split_text(self, text: str) -> List[Document]:
        """Split markdown file
        Args:
            text: Markdown file"""

        # Split the input text by newline character ("\n").
        lines = text.split("\n")
        # Final output
        lines_with_metadata: List[LineType] = []
        # Content and metadata of the chunk currently being processed
        current_content: List[str] = []
        current_metadata: Dict[str, str] = {}
        # Keep track of the nested header structure
        # header_stack: List[Dict[str, Union[int, str]]] = []
        header_stack: List[HeaderType] = []
        initial_metadata: Dict[str, str] = {}

        in_code_block = False

        for line in lines:
            stripped_line = line.strip()

            if stripped_line.startswith("```"):
                # code block in one row
                if stripped_line.count("```") >= 2:
                    in_code_block = False
                else:
                    in_code_block = not in_code_block

            if in_code_block:
                current_content.append(stripped_line)
                continue

            # Check each line against each of the header types (e.g., #, ##)
            for sep, name in self.headers_to_split_on:
                # Check if line starts with a header that we intend to split on
                if stripped_line.startswith(sep) and (
                    # Header with no text OR header is followed by space
                    # Both are valid conditions that sep is being used a header
                    len(stripped_line) == len(sep) or stripped_line[len(sep)] == " "
                ):
                    # Ensure we are tracking the header as metadata
                    if name is not None:
                        # Get the current header level
                        current_header_level = sep.count("#")

                        # Pop out headers of lower or same level from the stack
                        while (
                            header_stack
                            and header_stack[-1]["level"] >= current_header_level
                        ):
                            # We have encountered a new header
                            # at the same or higher level
                            popped_header = header_stack.pop()
                            # Clear the metadata for the
                            # popped header in initial_metadata
                            if popped_header["name"] in initial_metadata:
                                initial_metadata.pop(popped_header["name"])

                        # Push the current header to the stack
                        header: HeaderType = {
                            "level": current_header_level,
                            "name": name,
                            "data": stripped_line[len(sep) :].strip(),
                        }
                        header_stack.append(header)
                        # Update initial_metadata with the current header
                        initial_metadata[name] = header["data"]

                    # Add the previous line to the lines_with_metadata
                    # only if current_content is not empty
                    if current_content:
                        lines_with_metadata.append(
                            {
                                "content": "\n".join(current_content),
                                "metadata": current_metadata.copy(),
                            }
                        )
                        current_content.clear()

                    break
            else:
                if stripped_line:
                    current_content.append(stripped_line)
                elif current_content:
                    lines_with_metadata.append(
                        {
                            "content": "\n".join(current_content),
                            "metadata": current_metadata.copy(),
                        }
                    )
                    current_content.clear()

            current_metadata = initial_metadata.copy()

        if current_content:
            lines_with_metadata.append(
                {"content": "\n".join(current_content), "metadata": current_metadata}
            )

        # lines_with_metadata has each line with associated header metadata
        # aggregate these into chunks based on common metadata
        if not self.return_each_line:
            return self.aggregate_lines_to_chunks(lines_with_metadata)
        else:
            return [
                Document(page_content=chunk["content"], metadata=chunk["metadata"])
                for chunk in lines_with_metadata
            ]


class ElementType(TypedDict):
    """Element type as typed dict."""

    url: str
    xpath: str
    content: str
    metadata: Dict[str, str]


class HTMLHeaderTextSplitter:
    """
    Splitting HTML files based on specified headers.
    Requires lxml package.
    """

    def __init__(
        self,
        headers_to_split_on: List[Tuple[str, str]],
        return_each_element: bool = False,
    ):
        """Create a new HTMLHeaderTextSplitter.

        Args:
            headers_to_split_on: list of tuples of headers we want to track mapped to
                (arbitrary) keys for metadata. Allowed header values: h1, h2, h3, h4,
                h5, h6 e.g. [("h1", "Header 1"), ("h2", "Header 2)].
            return_each_element: Return each element w/ associated headers.
        """
        # Output element-by-element or aggregated into chunks w/ common headers
        self.return_each_element = return_each_element
        self.headers_to_split_on = sorted(headers_to_split_on)

    def aggregate_elements_to_chunks(
        self, elements: List[ElementType]
    ) -> List[Document]:
        """Combine elements with common metadata into chunks

        Args:
            elements: HTML element content with associated identifying info and metadata
        """
        aggregated_chunks: List[ElementType] = []

        for element in elements:
            if (
                aggregated_chunks
                and aggregated_chunks[-1]["metadata"] == element["metadata"]
            ):
                # If the last element in the aggregated list
                # has the same metadata as the current element,
                # append the current content to the last element's content
                aggregated_chunks[-1]["content"] += "  \n" + element["content"]
            else:
                # Otherwise, append the current element to the aggregated list
                aggregated_chunks.append(element)

        return [
            Document(page_content=chunk["content"], metadata=chunk["metadata"])
            for chunk in aggregated_chunks
        ]

    def split_text_from_url(self, url: str) -> List[Document]:
        """Split HTML from web URL

        Args:
            url: web URL
        """
        r = requests.get(url)
        return self.split_text_from_file(BytesIO(r.content))

    def split_text(self, text: str) -> List[Document]:
        """Split HTML text string

        Args:
            text: HTML text
        """
        return self.split_text_from_file(StringIO(text))

    def split_text_from_file(self, file: Any) -> List[Document]:
        """Split HTML file

        Args:
            file: HTML file
        """
        try:
            from lxml import etree
        except ImportError as e:
            raise ImportError(
                "Unable to import lxml, please install with `pip install lxml`."
            ) from e
        # use lxml library to parse html document and return xml ElementTree
        parser = etree.HTMLParser()
        tree = etree.parse(file, parser)

        # document transformation for "structure-aware" chunking is handled with xsl.
        # see comments in html_chunks_with_headers.xslt for more detailed information.
        xslt_path = (
            pathlib.Path(__file__).parent
            / "document_transformers/xsl/html_chunks_with_headers.xslt"
        )
        xslt_tree = etree.parse(xslt_path)
        transform = etree.XSLT(xslt_tree)
        result = transform(tree)
        result_dom = etree.fromstring(str(result))

        # create filter and mapping for header metadata
        header_filter = [header[0] for header in self.headers_to_split_on]
        header_mapping = dict(self.headers_to_split_on)

        # map xhtml namespace prefix
        ns_map = {"h": "http://www.w3.org/1999/xhtml"}

        # build list of elements from DOM
        elements = []
        for element in result_dom.findall("*//*", ns_map):
            if element.findall("*[@class='headers']") or element.findall(
                "*[@class='chunk']"
            ):
                elements.append(
                    ElementType(
                        url=file,
                        xpath="".join(
                            [
                                node.text
                                for node in element.findall("*[@class='xpath']", ns_map)
                            ]
                        ),
                        content="".join(
                            [
                                node.text
                                for node in element.findall("*[@class='chunk']", ns_map)
                            ]
                        ),
                        metadata={
                            # Add text of specified headers to metadata using header
                            # mapping.
                            header_mapping[node.tag]: node.text
                            for node in filter(
                                lambda x: x.tag in header_filter,
                                element.findall("*[@class='headers']/*", ns_map),
                            )
                        },
                    )
                )

        if not self.return_each_element:
            return self.aggregate_elements_to_chunks(elements)
        else:
            return [
                Document(page_content=chunk["content"], metadata=chunk["metadata"])
                for chunk in elements
            ]


# should be in newer Python versions (3.10+)
# @dataclass(frozen=True, kw_only=True, slots=True)
@dataclass(frozen=True)
class Tokenizer:
    """Tokenizer data class."""

    chunk_overlap: int
    """Overlap in tokens between chunks"""
    tokens_per_chunk: int
    """Maximum number of tokens per chunk"""
    decode: Callable[[List[int]], str]
    """ Function to decode a list of token ids to a string"""
    encode: Callable[[str], List[int]]
    """ Function to encode a string to a list of token ids"""


def split_text_on_tokens(*, text: str, tokenizer: Tokenizer) -> List[str]:
    """Split incoming text and return chunks using tokenizer."""
    splits: List[str] = []
    input_ids = tokenizer.encode(text)
    start_idx = 0
    cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
    chunk_ids = input_ids[start_idx:cur_idx]
    while start_idx < len(input_ids):
        splits.append(tokenizer.decode(chunk_ids))
        start_idx += tokenizer.tokens_per_chunk - tokenizer.chunk_overlap
        cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]
    return splits


class TokenTextSplitter(TextSplitter):
    """Splitting text to tokens using model tokenizer."""

    def __init__(
        self,
        encoding_name: str = "gpt2",
        model_name: Optional[str] = None,
        allowed_special: Optional[Union[Literal["all"], AbstractSet[str]]] = None,
        disallowed_special: Union[Literal["all"], Collection[str]] = "all",
        **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(**kwargs)
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "Could not import tiktoken python package. "
                "This is needed in order to for TokenTextSplitter. "
                "Please install it with `pip install tiktoken`."
            )

        allowed_special = allowed_special or set()

        if model_name is not None:
            enc = tiktoken.encoding_for_model(model_name)
        else:
            enc = tiktoken.get_encoding(encoding_name)
        self._tokenizer = enc
        self._allowed_special = allowed_special
        self._disallowed_special = disallowed_special

    def split_text(self, text: str) -> List[str]:
        def _encode(_text: str) -> List[int]:
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


class SentenceTransformersTokenTextSplitter(TextSplitter):
    """Splitting text to tokens using sentence model tokenizer."""

    def __init__(
        self,
        chunk_overlap: int = 50,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        tokens_per_chunk: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(**kwargs, chunk_overlap=chunk_overlap)

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Could not import sentence_transformer python package. "
                "This is needed in order to for SentenceTransformersTokenTextSplitter. "
                "Please install it with `pip install sentence-transformers`."
            )

        self.model_name = model_name
        self._model = SentenceTransformer(self.model_name)
        self.tokenizer = self._model.tokenizer
        self._initialize_chunk_configuration(tokens_per_chunk=tokens_per_chunk)

    def _initialize_chunk_configuration(
        self, *, tokens_per_chunk: Optional[int]
    ) -> None:
        self.maximum_tokens_per_chunk = cast(int, self._model.max_seq_length)

        if tokens_per_chunk is None:
            self.tokens_per_chunk = self.maximum_tokens_per_chunk
        else:
            self.tokens_per_chunk = tokens_per_chunk

        if self.tokens_per_chunk > self.maximum_tokens_per_chunk:
            raise ValueError(
                f"The token limit of the models '{self.model_name}'"
                f" is: {self.maximum_tokens_per_chunk}."
                f" Argument tokens_per_chunk={self.tokens_per_chunk}"
                f" > maximum token limit."
            )

    def split_text(self, text: str) -> List[str]:
        def encode_strip_start_and_stop_token_ids(text: str) -> List[int]:
            return self._encode(text)[1:-1]

        tokenizer = Tokenizer(
            chunk_overlap=self._chunk_overlap,
            tokens_per_chunk=self.tokens_per_chunk,
            decode=self.tokenizer.decode,
            encode=encode_strip_start_and_stop_token_ids,
        )

        return split_text_on_tokens(text=text, tokenizer=tokenizer)

    def count_tokens(self, *, text: str) -> int:
        return len(self._encode(text))

    _max_length_equal_32_bit_integer: int = 2**32

    def _encode(self, text: str) -> List[int]:
        token_ids_with_start_and_end_token_ids = self.tokenizer.encode(
            text,
            max_length=self._max_length_equal_32_bit_integer,
            truncation="do_not_truncate",
        )
        return token_ids_with_start_and_end_token_ids


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


class RecursiveCharacterTextSplitter(CharacterTextSplitter):
    """Splitting text by recursively look at characters.

    Recursively tries to split by different characters to find one
    that works.
    """

    def __init__(
        self,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        is_separator_regex: bool = False,
        **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter.

        Args:
            separators: List of separators to try
            keep_separator: Whether to keep the separator in the chunks
            is_separator_regex: Whether the separator is a regex
        """
        super().__init__(
            separator="",
            is_separator_regex=is_separator_regex,
            keep_separator=keep_separator,
            **kwargs,
        )
        separators = separators or ["\n\n", "\n", " ", ""]

        # If the separator is a regex, we don't need to escape it.
        if not self._is_separator_regex:
            separators = [re.escape(separator) for separator in separators]
        # Join separators.
        self._separator = "|".join(separators)
        # Capture separators.
        if self._separator:
            self._separator = f"({self._separator})"

    @classmethod
    def from_language(
        cls, language: Language, **kwargs: Any
    ) -> RecursiveCharacterTextSplitter:
        separators = cls.get_separators_for_language(language)
        return cls(separators=separators, is_separator_regex=True, **kwargs)

    @staticmethod
    def get_separators_for_language(language: Language) -> List[str]:
        if language == Language.CPP:
            return [
                # Split along class definitions
                # "\nclass ",
                # Split along function definitions
                # "\nvoid ",
                # "\nint ",
                # "\nfloat ",
                # "\ndouble ",
                # Split along control flow statements
                # "\nif ",
                # "\nfor ",
                # "\nwhile ",
                # "\nswitch ",
                # "\ncase ",
                # Split by the normal type of lines
                "\n+",
                " +",
            ]
        elif language == Language.GO:
            return [
                # Split along function definitions
                # "\nfunc ",
                # "\nvar ",
                # "\nconst ",
                # "\ntype ",
                # Split along control flow statements
                # "\nif ",
                # "\nfor ",
                # "\nswitch ",
                # "\ncase ",
                # Split by the normal type of lines
                "\n+",
                " +",
            ]
        elif language == Language.JAVA:
            return [
                # Split along class definitions
                # "\nclass ",
                # Split along method definitions
                # "\npublic ",
                # "\nprotected ",
                # "\nprivate ",
                # "\nstatic ",
                # Split along control flow statements
                # "\nif ",
                # "\nfor ",
                # "\nwhile ",
                # "\nswitch ",
                # "\ncase ",
                # Split by the normal type of lines
                "\n+",
                " +",
            ]
        elif language == Language.KOTLIN:
            return [
                # # Split along class definitions
                # "\nclass ",
                # # Split along method definitions
                # "\npublic ",
                # "\nprotected ",
                # "\nprivate ",
                # "\ninternal ",
                # "\ncompanion ",
                # "\nfun ",
                # "\nval ",
                # "\nvar ",
                # # Split along control flow statements
                # "\nif ",
                # "\nfor ",
                # "\nwhile ",
                # "\nwhen ",
                # "\ncase ",
                # "\nelse ",
                # Split by the normal type of lines
                "\n+",
                " +",
            ]
        elif language == Language.JS:
            return [
                # Split along function definitions
                # "\nfunction ",
                # "\nconst ",
                # "\nlet ",
                # "\nvar ",
                # "\nclass ",
                # Split along control flow statements
                # "\nif ",
                # "\nfor ",
                # "\nwhile ",
                # "\nswitch ",
                # "\ncase ",
                # "\ndefault ",
                # Split by the normal type of lines
                "\n+",
                " +",
            ]
        elif language == Language.TS:
            return [
                # "\nenum ",
                # "\ninterface ",
                # "\nnamespace ",
                # "\ntype ",
                # # Split along class definitions
                # "\nclass ",
                # # Split along function definitions
                # "\nfunction ",
                # "\nconst ",
                # "\nlet ",
                # "\nvar ",
                # # Split along control flow statements
                # "\nif ",
                # "\nfor ",
                # "\nwhile ",
                # "\nswitch ",
                # "\ncase ",
                # "\ndefault ",
                # Split by the normal type of lines
                "\n+",
                " +",
            ]
        elif language == Language.PHP:
            return [
                # Split along function definitions
                # "\nfunction ",
                # Split along class definitions
                # "\nclass ",
                # Split along control flow statements
                # "\nif ",
                # "\nforeach ",
                # "\nwhile ",
                # "\ndo ",
                # "\nswitch ",
                # "\ncase ",
                # Split by the normal type of lines
                "\n+",
                " +",
            ]
        elif language == Language.PROTO:
            return [
                # Split along message definitions
                # "\nmessage ",
                # Split along service definitions
                # "\nservice ",
                # Split along enum definitions
                # "\nenum ",
                # Split along option definitions
                # "\noption ",
                # Split along import statements
                # "\nimport ",
                # Split along syntax declarations
                # "\nsyntax ",
                # Split by the normal type of lines
                "\n+",
                " +",
            ]
        elif language == Language.PYTHON:
            return [
                # First, try to split along class definitions
                # "\nclass ",
                # "\ndef ",
                # "\n\tdef ",
                # Now split by the normal type of lines
                "\n+",
                " +",
            ]
        elif language == Language.RST:
            return [
                # Split along section titles
                "\n=+\n",
                "\n-+\n",
                "\n\\*+\n",
                # Split along directive markers
                "\n\n.. *\n\n",
                "\n- .\n",
                # Split by the normal type of lines
                "\n+",
                " +",
            ]
        elif language == Language.RUBY:
            return [
                # Split along method definitions
                # "\ndef ",
                # "\nclass ",
                # Split along control flow statements
                # "\nif ",
                # "\nunless ",
                # "\nwhile ",
                # "\nfor ",
                # "\ndo ",
                # "\nbegin ",
                # "\nrescue ",
                # Split by the normal type of lines
                "\n+",
                " +",
            ]
        elif language == Language.RUST:
            return [
                # Split along function definitions
                # "\nfn ",
                # "\nconst ",
                # "\nlet ",
                # Split along control flow statements
                # "\nif ",
                # "\nwhile ",
                # "\nfor ",
                # "\nloop ",
                # "\nmatch ",
                # "\nconst ",
                # Split by the normal type of lines
                "\n+",
                " +",
            ]
        elif language == Language.SCALA:
            return [
                # Split along class definitions
                # "\nclass ",
                # "\nobject ",
                # Split along method definitions
                # "\ndef ",
                # "\nval ",
                # "\nvar ",
                # Split along control flow statements
                # "\nif ",
                # "\nfor ",
                # "\nwhile ",
                # "\nmatch ",
                # "\ncase ",
                # Split by the normal type of lines
                "\n+",
                " +",
            ]
        elif language == Language.SWIFT:
            return [
                # Split along function definitions
                # "\nfunc ",
                # Split along class definitions
                # "\nclass ",
                # "\nstruct ",
                # "\nenum ",
                # Split along control flow statements
                # "\nif ",
                # "\nfor ",
                # "\nwhile ",
                # "\ndo ",
                # "\nswitch ",
                # "\ncase ",
                # Split by the normal type of lines
                "\n+",
                " +",
            ]
        elif language == Language.MARKDOWN:
            return [
                # First, try to split along Markdown headings (starting with level 2)
                "\n#{1,6} ",
                # Note the alternative syntax for headings (below) is not handled here
                # Heading level 2
                # ---------------
                # End of code block
                "```\n",
                # Horizontal lines
                "\n\\*\\*\\*+\n",
                "\n---+\n",
                "\n___+\n",
                # Note that this splitter doesn't handle horizontal lines defined
                # by *three or more* of ***, ---, or ___, but this is not handled
                "\n+",
                " +",
            ]
        elif language == Language.LATEX:
            return [
                # First, try to split along Latex sections
                "\n\\\\chapter{",
                "\n\\\\section{",
                "\n\\\\subsection{",
                "\n\\\\subsubsection{",
                # Now split by environments
                "\n\\\\begin{enumerate}",
                "\n\\\\begin{itemize}",
                "\n\\\\begin{description}",
                "\n\\\\begin{list}",
                "\n\\\\begin{quote}",
                "\n\\\\begin{quotation}",
                "\n\\\\begin{verse}",
                "\n\\\\begin{verbatim}",
                # Now split by math environments
                "\n\\\begin/{align/}",
                "\$+",
                # Now split by the normal type of lines
            ]
        elif language == Language.HTML:
            return [
                # First, try to split along HTML tags
                # "<body",
                # "<div",
                # "<p",
                # "<br",
                # "<li",
                # "<h1",
                # "<h2",
                # "<h3",
                # "<h4",
                # "<h5",
                # "<h6",
                # "<span",
                # "<table",
                # "<tr",
                # "<td",
                # "<th",
                # "<ul",
                # "<ol",
                # "<header",
                # "<footer",
                # "<nav",
                # # Head
                # "<head",
                # "<style",
                # "<script",
                # "<meta",
                # "<title",
                "\n+",
                " +",
            ]
        elif language == Language.SOL:
            return [
                # Split along compiler information definitions
                # "\npragma ",
                # "\nusing ",
                # Split along contract definitions
                # "\ncontract ",
                # "\ninterface ",
                # "\nlibrary ",
                # Split along method definitions
                # "\nconstructor ",
                # "\ntype ",
                # "\nfunction ",
                # "\nevent ",
                # "\nmodifier ",
                # "\nerror ",
                # "\nstruct ",
                # "\nenum ",
                # Split along control flow statements
                # "\nif ",
                # "\nfor ",
                # "\nwhile ",
                # "\ndo while ",
                # "\nassembly ",
                # Split by the normal type of lines
                "\n+",
                " +",
            ]
        elif language == Language.CSHARP:
            return [
                # "\ninterface ",
                # "\nenum ",
                # "\nimplements ",
                # "\ndelegate ",
                # "\nevent ",
                # Split along class definitions
                # "\nclass ",
                # "\nabstract ",
                # Split along method definitions
                # "\npublic ",
                # "\nprotected ",
                # "\nprivate ",
                # "\nstatic ",
                # "\nreturn ",
                # Split along control flow statements
                # "\nif ",
                # "\ncontinue ",
                # "\nfor ",
                # "\nforeach ",
                # "\nwhile ",
                # "\nswitch ",
                # "\nbreak ",
                # "\ncase ",
                # "\nelse ",
                # Split by exceptions
                # "\ntry ",
                # "\nthrow ",
                # "\nfinally ",
                # "\ncatch ",
                # Split by the normal type of lines
                "\n+",
                " +",
            ]
        elif language == Language.COBOL:
            return [
                # # Split along divisions
                # "\nIDENTIFICATION DIVISION.",
                # "\nENVIRONMENT DIVISION.",
                # "\nDATA DIVISION.",
                # "\nPROCEDURE DIVISION.",
                # # Split along sections within DATA DIVISION
                # "\nWORKING-STORAGE SECTION.",
                # "\nLINKAGE SECTION.",
                # "\nFILE SECTION.",
                # # Split along sections within PROCEDURE DIVISION
                # "\nINPUT-OUTPUT SECTION.",
                # # Split along paragraphs and common statements
                # "\nOPEN ",
                # "\nCLOSE ",
                # "\nREAD ",
                # "\nWRITE ",
                # "\nIF ",
                # "\nELSE ",
                # "\nMOVE ",
                # "\nPERFORM ",
                # "\nUNTIL ",
                # "\nVARYING ",
                # "\nACCEPT ",
                # "\nDISPLAY ",
                # "\nSTOP RUN.",
                # Split by the normal type of lines
                "\n+",
                " +",
            ]
        else:
            logging.warning(
                f"Language {language} not supported. Using default splitters."
            )
            return [
                "\n+",
                " +",
            ]


class TokenCharacterTextSplitter(TextSplitter, ABC):
    def __init__(self, separator: str = "\n\n", **kwargs: Any) -> None:
        """Initialize the NLTK splitter."""
        super().__init__(**kwargs)
        self._separator = separator

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        splits_str = self._text_split(text)
        splits = [
            SplittedText(text=self._separator, is_separator=True)
            if i % 2 != 0
            else SplittedText(text=splits_str[i // 2], is_separator=False)
            for i in range(len(splits_str) * 2 - 1)
        ]
        return self._merge_splits(splits)

    @abstractmethod
    def _text_split(self, text: str) -> List[str]:
        pass


class NLTKTextSplitter(TokenCharacterTextSplitter):
    """Splitting text using NLTK package."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the NLTK splitter."""
        super().__init__(**kwargs)
        try:
            from nltk.tokenize import sent_tokenize

            self._tokenizer = sent_tokenize
        except ImportError:
            raise ImportError(
                "NLTK is not installed, please install it with `pip install nltk`."
            )

    def _text_split(self, text: str) -> List[str]:
        return self._tokenizer(text)


class SpacyTextSplitter(TokenCharacterTextSplitter):
    """Splitting text using Spacy package.


    Per default, Spacy's `en_core_web_sm` model is used. For a faster, but
    potentially less accurate splitting, you can use `pipeline='sentencizer'`.
    """

    def __init__(self, pipeline: str = "en_core_web_sm", **kwargs: Any) -> None:
        """Initialize the spacy text splitter."""
        super().__init__(**kwargs)
        self._tokenizer = _make_spacy_pipeline_for_splitting(pipeline)

    def _text_split(self, text: str) -> List[str]:
        return [s.text for s in self._tokenizer(text).sents]


# For backwards compatibility
class PythonCodeTextSplitter(RecursiveCharacterTextSplitter):
    """Attempts to split the text along Python syntax."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a PythonCodeTextSplitter."""
        separators = self.get_separators_for_language(Language.PYTHON)
        super().__init__(separators=separators, is_separator_regex=True, **kwargs)


class MarkdownTextSplitter(RecursiveCharacterTextSplitter):
    """Attempts to split the text along Markdown-formatted headings."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a MarkdownTextSplitter."""
        separators = self.get_separators_for_language(Language.MARKDOWN)
        super().__init__(separators=separators, **kwargs)


class LatexTextSplitter(RecursiveCharacterTextSplitter):
    """Attempts to split the text along Latex-formatted layout elements."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a LatexTextSplitter."""
        separators = self.get_separators_for_language(Language.LATEX)
        super().__init__(separators=separators, **kwargs)
