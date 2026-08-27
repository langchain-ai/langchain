"""DocumentFilter that uses an LLM chain to extract the relevant parts of documents."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from pydantic import ConfigDict
from typing_extensions import override

from langchain_classic.chains.llm import LLMChain
from langchain_classic.retrievers.document_compressors.chain_extract_prompt import (
    prompt_template,
)


def default_get_input(query: str, doc: Document) -> dict[str, Any]:
    """Return the compression chain input."""
    return {"question": query, "context": doc.page_content}


class NoOutputParser(BaseOutputParser[str]):
    """Parse outputs that could return a null string of some sort."""

    no_output_str: str = "NO_OUTPUT"

    @override
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
    """LLM Chain Extractor.

    Document compressor that uses an LLM chain to extract
    the relevant parts of documents.
    """

    llm_chain: Runnable
    """LLM wrapper to use for compressing documents."""

    get_input: Callable[[str, Document], dict] = default_get_input
    """Callable for constructing the chain input from the query and a Document."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        """Compress page content of raw documents."""
        compressed_docs = []
        for doc in documents:
            _input = self.get_input(query, doc)
            output_ = self.llm_chain.invoke(_input, config={"callbacks": callbacks})
            if isinstance(self.llm_chain, LLMChain):
                output = output_[self.llm_chain.output_key]
                if self.llm_chain.prompt.output_parser is not None:
                    output = self.llm_chain.prompt.output_parser.parse(output)
            else:
                output = output_
            if len(output) == 0:
                continue
            compressed_docs.append(
                Document(page_content=cast("str", output), metadata=doc.metadata),
            )
        return compressed_docs

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        """Compress page content of raw documents asynchronously."""
        inputs = [self.get_input(query, doc) for doc in documents]
        outputs = await self.llm_chain.abatch(inputs, {"callbacks": callbacks})
        compressed_docs = []
        for i, doc in enumerate(documents):
            if len(outputs[i]) == 0:
                continue
            compressed_docs.append(
                Document(page_content=outputs[i], metadata=doc.metadata),
            )
        return compressed_docs

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: PromptTemplate | None = None,
        get_input: Callable[[str, Document], str] | None = None,
        llm_chain_kwargs: dict | None = None,  # noqa: ARG003
    ) -> LLMChainExtractor:
        """Initialize from LLM."""
        _prompt = prompt if prompt is not None else _get_default_chain_prompt()
        _get_input = get_input if get_input is not None else default_get_input
        if _prompt.output_parser is not None:
            parser = _prompt.output_parser
        else:
            parser = StrOutputParser()
        llm_chain = _prompt | llm | parser
        return cls(llm_chain=llm_chain, get_input=_get_input)
