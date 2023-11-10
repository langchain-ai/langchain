"""
    DocumentCompressor that uses an optimized encoded LLM chain
    to extract the relevant parts of documents.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any, Callable, Dict, List, Optional, Sequence

from langchain.callbacks.manager import Callbacks
from langchain.prompts import PromptTemplate
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.retrievers.document_compressors.encoded_chain_extract_prompt import (
    prompt_template,
)
from langchain.schema import BaseOutputParser, Document, OutputParserException
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.runnable import RunnableConfig, RunnableSerializable
from langchain.text_splitter import _make_spacy_pipeline_for_splitting

NO_OUTPUT_STR: str = "NO_OUTPUT"


class SequenceListParser(BaseOutputParser[List[int]]):
    """Parse outputs that contain a sequence list"""

    def _expand_range(self, range_str: str) -> List[int]:
        """Expand a range string into a list of numbers."""
        start, end = range_str.split("-")
        if int(start) > int(end):
            raise ValueError(f"Invalid range: {range_str}")
        return list(range(int(start), int(end) + 1))

    def parse(self, text: str) -> List[int]:
        cleaned_text = text.strip()
        if not cleaned_text or cleaned_text == NO_OUTPUT_STR:
            return []
        returned_list = [i.strip() for i in cleaned_text.split(",")]
        sequence_set = set()
        for i in returned_list:
            if "-" in i:
                try:
                    sequence_set.update(self._expand_range(i))
                except ValueError as e:
                    raise OutputParserException(f"Invalid range: {i}") from e
            else:
                try:
                    sequence_set.add(int(i))
                except ValueError as e:
                    raise OutputParserException(f"Invalid sequence number: {i}") from e
        return list(sequence_set)


def _split_paragraphs(text: str) -> List[str]:
    """Split the input text into paragraphs using "\n\n" as the delimiter."""

    # Split based on a newline, followed by spaces/tabs, then another newline.
    paras = re.split(r"\n[ \t]*\n", text)
    return [para.strip() for para in paras if para.strip()]


def get_default_tokenizer() -> Any:
    return _make_spacy_pipeline_for_splitting(pipeline="en_core_web_sm")


def number_sequences(text: str, len: int = 1, tokenizer: Any = None) -> str:
    """
    Number the sequences in a given text, preserving paragraph structure.
    The default number of sentences per sequence is 1.

    Args:
        text: the text to number.
        len: the number of sentences per sequence.

    Returns:
        the text with numbered sequences.
    """
    numbered_text = []
    count = 0

    if tokenizer is None:
        tokenizer = get_default_tokenizer()
    paragraphs = _split_paragraphs(text)
    for paragraph in paragraphs:
        sentences = [sent.text for sent in tokenizer(paragraph).sents]
        for i, sentence in enumerate(sentences):
            num = count // len + 1
            number_prefix = f"#|{num}|#" if count % len == 0 else ""
            sentence = f"{number_prefix} {sentence}"
            count += 1
            sentences[i] = sentence
        numbered_paragraph = " ".join(sentences)
        numbered_text.append(numbered_paragraph)

    return "  \n\n  ".join(numbered_text)


def default_get_input(query: str, context: str) -> Dict[str, str]:
    """Return the compression chain input."""
    return {"question": query, "context": context}


def _get_default_chain_prompt() -> PromptTemplate:
    template = prompt_template.format(no_output_str=NO_OUTPUT_STR)
    return PromptTemplate(
        template=template,
        input_variables=["question", "context"],
    )


def extract_numbered_sequences(text: str, sequence_list: List[int]) -> str:
    """
    Extract specified sequences from a numbered text,
    preserving paragraph structure.

    Args:
        text: the text to extract from.
        sequence_list: the list of sequence numbers to extract.

    Returns:
        the extracted sequences from the text.
    """

    if not sequence_list:
        return ""
    sequence_pattern = re.compile(r"#\|(\d+)\|# ((?:(?!#\|).)+)")
    paragraphs = _split_paragraphs(text)

    extracted_paragraphs = []

    for paragraph in paragraphs:
        paragraph = paragraph.replace("\n", "\t")
        sequences_with_numbers = sequence_pattern.findall(paragraph)

        if extracted_sequences := [
            sequence
            for num, sequence in sequences_with_numbers
            if int(num) in sequence_list
        ]:
            extracted_paragraphs.append(" ".join(extracted_sequences))

    return "\n\n".join(extracted_paragraphs)


class LLMEncodedChainExtractor(BaseDocumentCompressor):
    """
    Document compressor that uses an LLM chain to extract
    the relevant parts of documents.
    The relevant parts are encoded as sequence numbers
    to optimize the extraction process.
    """

    llm_chain: RunnableSerializable[dict[str, str], List[int]]
    """LLM wrapper to use for compressing documents."""

    get_input: Callable[[str, str], dict] = default_get_input
    """Callable for constructing the chain input from the query and a Document."""

    tokenizer: Any
    """Tokenizer to use for splitting text into sentences."""

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Compress page content of raw documents.

        Args:
            documents: list of raw Documents
            query: user query
            callbacks: callbacks to run/attach

        Returns:
            list of compressed Documents
        """
        compressed_docs: List[Document] = []
        _config = RunnableConfig(callbacks=callbacks)
        for doc in documents:
            doc_content = number_sequences(doc.page_content, tokenizer=self.tokenizer)
            _input = self.get_input(query, doc_content)
            sequence_list = self.llm_chain.invoke(_input, _config)
            assert isinstance(sequence_list, list)
            if len(sequence_list) == 0:
                continue
            compressed_docs.append(
                Document(
                    page_content=extract_numbered_sequences(doc_content, sequence_list),
                    metadata=doc.metadata,
                )
            )
        return compressed_docs

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Compress page content of raw documents asynchronously.

        Args:
            documents: list of raw Documents
            query: user query
            callbacks: callbacks to run/attach

        Returns:
            list of compressed Documents
        """
        _config = RunnableConfig(callbacks=callbacks)
        numbered_sequence_docs = [
            Document(
                page_content=number_sequences(
                    doc.page_content, tokenizer=self.tokenizer
                ),
                metadata=doc.metadata,
            )
            for doc in documents
        ]
        sequence_lists = await asyncio.gather(
            *[
                self.llm_chain.ainvoke(self.get_input(query, doc.page_content), _config)
                for doc in numbered_sequence_docs
            ]
        )
        return [
            Document(
                page_content=extract_numbered_sequences(
                    doc.page_content, sequence_lists[i]
                ),
                metadata=doc.metadata,
            )
            for i, doc in enumerate(numbered_sequence_docs)
            if len(sequence_lists[i]) != 0
        ]

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        get_input: Optional[Callable[[str, str], dict[str, str]]] = None,
        pipeline: str = "en_core_web_sm",
    ) -> LLMEncodedChainExtractor:
        """Initialize from LLM.

        Args:
            llm: the LLM to use
            get_input: callable for constructing the chain input from the query
                        and context
            llm_chain_kwargs: kwargs to pass to the LLM chain

        Returns:
            the initialized LLMEncodedChainExtractor
        """
        _prompt = _get_default_chain_prompt()
        _get_input = get_input if get_input is not None else default_get_input
        _output_parser = SequenceListParser()
        tokenizer = (
            _make_spacy_pipeline_for_splitting(pipeline=pipeline)
            if pipeline
            else get_default_tokenizer()
        )
        llm_chain = _prompt | llm | _output_parser
        return cls(llm_chain=llm_chain, get_input=_get_input, tokenizer=tokenizer)
