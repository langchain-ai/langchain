"""DocumentFilter that uses an LLM chain to extract the relevant parts of documents."""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, Optional, Sequence, cast

from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate

from langchain.chains.llm import LLMChain
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.retrievers.document_compressors.chain_extract_prompt import (
    prompt_template,
)


def default_get_input(query: str, doc: Document) -> Dict[str, Any]:
    """Return the compression chain input."""
    return {"question": query, "context": doc.page_content}


class NoOutputParser(BaseOutputParser[str]):
    """Parse outputs that could return a null string of some sort."""

    no_output_str: str = "NO_OUTPUT"

    def parse(self, text: str) -> str:
        cleaned_text = text.strip()
        if cleaned_text == self.no_output_str:
            return ""
        return cleaned_text


def _get_default_chain_prompt() -> PromptTemplate:
    output_parser = NoOutputParser()
    template = prompt_template.format(no_output_str=output_parser.no_output_str)
    return PromptTemplate(
        template=template,
        input_variables=["question", "context"],
        output_parser=output_parser,
    )


class LLMChainExtractor(BaseDocumentCompressor):
    """Document compressor that uses an LLM chain to extract
    the relevant parts of documents."""

    llm_chain: LLMChain
    """LLM wrapper to use for compressing documents."""

    get_input: Callable[[str, Document], dict] = default_get_input
    """Callable for constructing the chain input from the query and a Document."""

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Compress page content of raw documents."""
        compressed_docs = []
        for doc in documents:
            _input = self.get_input(query, doc)
            output = self.llm_chain.predict_and_parse(**_input, callbacks=callbacks)
            if len(output) == 0:
                continue
            compressed_docs.append(
                Document(page_content=cast(str, output), metadata=doc.metadata)
            )
        return compressed_docs

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Compress page content of raw documents asynchronously."""
        outputs = await asyncio.gather(
            *[
                self.llm_chain.apredict_and_parse(
                    **self.get_input(query, doc), callbacks=callbacks
                )
                for doc in documents
            ]
        )
        compressed_docs = []
        for i, doc in enumerate(documents):
            if len(outputs[i]) == 0:
                continue
            compressed_docs.append(
                Document(page_content=outputs[i], metadata=doc.metadata)  # type: ignore[arg-type]
            )
        return compressed_docs

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: Optional[PromptTemplate] = None,
        get_input: Optional[Callable[[str, Document], str]] = None,
        llm_chain_kwargs: Optional[dict] = None,
    ) -> LLMChainExtractor:
        """Initialize from LLM."""
        _prompt = prompt if prompt is not None else _get_default_chain_prompt()
        _get_input = get_input if get_input is not None else default_get_input
        llm_chain = LLMChain(llm=llm, prompt=_prompt, **(llm_chain_kwargs or {}))
        return cls(llm_chain=llm_chain, get_input=_get_input)  # type: ignore[arg-type]
