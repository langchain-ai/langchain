"""DocumentFilter that uses an LLM chain to extract the relevant parts of documents."""
from typing import Any, Callable, Dict, List, Optional

from langchain import LLMChain, PromptTemplate
from langchain.retrievers.document_filters.base import (
    BaseDocumentFilter,
    _RetrievedDocument,
)
from langchain.retrievers.document_filters.compression_chain_prompt import (
    prompt_template,
)
from langchain.schema import BaseLanguageModel, BaseOutputParser, Document


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


class LLMChainDocumentCompressor(BaseDocumentFilter):
    llm_chain: LLMChain
    """LLM wrapper to use for compressing documents."""

    get_input: Callable[[str, Document], dict] = default_get_input
    """Callable for constructing the chain input from the query and a Document."""

    def filter(
        self, docs: List[_RetrievedDocument], query: str
    ) -> List[_RetrievedDocument]:
        """Compress page content of raw documents."""
        compressed_docs = []
        for doc in docs:
            _input = self.get_input(query, doc)
            output = self.llm_chain.predict_and_parse(**_input)
            if len(output) == 0:
                continue
            compressed_docs.append(
                _RetrievedDocument(page_content=output, metadata=doc.metadata)
            )
        return compressed_docs

    async def afilter(
        self, docs: List[_RetrievedDocument], query: str
    ) -> List[_RetrievedDocument]:
        raise NotImplementedError

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: Optional[PromptTemplate] = None,
        get_input: Optional[Callable[[str, Document], str]] = None,
    ) -> "LLMChainDocumentCompressor":
        """Initialize from LLM."""
        _prompt = prompt if prompt is not None else _get_default_chain_prompt()
        _get_input = get_input if get_input is not None else default_get_input
        llm_chain = LLMChain(llm=llm, prompt=_prompt)
        return cls(llm_chain=llm_chain, get_input=_get_input)
