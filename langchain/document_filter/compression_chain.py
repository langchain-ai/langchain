""""""
from typing import Any, Callable, Dict, List, Optional

from langchain import LLMChain, PromptTemplate
from langchain.document_filter.base import BaseDocumentFilter
from langchain.document_filter.compression_chain_prompt import prompt_template
from langchain.schema import BaseLanguageModel, Document


def default_get_input(query: str, doc: Document) -> Dict[str, Any]:
    """Return the compression chain input."""
    return {"question": query, "context": doc.page_content}


class LLMChainCompressor(BaseDocumentFilter):
    llm_chain: LLMChain
    """LLM wrapper to use for compressing documents."""

    get_input: Callable[[str, Document], dict] = default_get_input
    """Callable for constructing the chain input from the query and a Document."""

    def filter(self, docs: List[Document], query: str) -> List[Document]:
        """Compress page content of raw documents."""
        compressed_docs = []
        for doc in docs:
            _input = self.get_input(query, doc)
            output = self.llm_chain.predict(**_input)
            if len(output) == 0:
                continue
            compressed_docs.append(Document(page_content=output, metadata=doc.metadata))
        return compressed_docs

    def afilter(self, docs: List[Document], query: str) -> List[Document]:
        raise NotImplementedError

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: Optional[PromptTemplate] = None,
        get_input: Optional[Callable[[str, Document], str]] = None,
    ) -> "LLMChainCompressor":
        """Initialize from LLM."""
        _prompt = prompt if prompt is not None else _get_default_chain_prompt()
        _get_input = get_input if get_input is not None else default_get_input
        llm_chain = LLMChain(llm=llm, prompt=_prompt)
        return cls(llm_chain=llm_chain, get_input=_get_input)


def _get_default_chain_prompt() -> PromptTemplate:
    return PromptTemplate(
        template=prompt_template, input_variables=["question", "context"]
    )
