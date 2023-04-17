""""""
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Extra

from langchain import LLMChain, PromptTemplate
from langchain.schema import BaseLanguageModel, BaseRetriever, Document


def default_get_input(query: str, doc: Document) -> Dict[str, Any]:
    """Return the compression chain input."""
    return {"question": query, "context": doc.page_content}


class BaseDocumentFilter(BaseModel, ABC):
    @abstractmethod
    def filter(self, docs: List[Document], query: str) -> List[Document]:
        """Filter down documents."""

    @abstractmethod
    def afilter(self, docs: List[Document], query: str) -> List[Document]:
        """Filter down documents."""


class LLMChainFilter(BaseDocumentFilter):
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
    ) -> "LLMChainFilter":
        """Initialize from LLM."""
        _prompt = prompt if prompt is not None else _get_default_chain_prompt()
        _get_input = get_input if get_input is not None else default_get_input
        llm_chain = LLMChain(llm=llm, prompt=_prompt)
        return cls(llm_chain=llm_chain, get_input=_get_input)


def _get_default_chain_prompt() -> PromptTemplate:
    template = """Given the following question and context, extract any part of the context *as is* that is relevant to answer the question.

    > Question: {question}
    > Context:
    >>>
    {context}
    >>>
    Extracted relevant parts:"""

    return PromptTemplate(template=template, input_variables=["question", "context"])


class ContextualCompressionRetriever(BaseRetriever, BaseModel):
    """"""

    base_filter: BaseDocumentFilter
    """Filter for filtering documents."""

    base_retriever: BaseRetriever
    """Base Retriever to use for getting relevant documents."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            List of relevant documents
        """
        docs = self.base_retriever.get_relevant_documents(query)
        compressed_docs = self.base_filter.filter(docs, query)
        return compressed_docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            List of relevant documents
        """
        docs = await self.base_retriever.aget_relevant_documents(query)
        compressed_docs = self.base_filter.afilter(docs, query)
        return compressed_docs
