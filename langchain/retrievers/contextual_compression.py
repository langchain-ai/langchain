""""""
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Extra

from langchain import LLMChain, PromptTemplate
from langchain.schema import BaseLanguageModel, BaseRetriever, Document


def _get_default_chain_prompt() -> PromptTemplate:
    template = """Given the following question and context, extract any part of the context *as is* that is relevant to answer the question.

    > Question: {question}
    > Context:
    >>>
    {context}
    >>>
    Extracted relevant parts:"""

    return PromptTemplate(template=template, input_variables=["question", "context"])


def default_get_input(query: str, doc: Document) -> Dict[str, Any]:
    """Return the compression chain input."""
    return {"question": query, "context": doc.page_content}


class ContextualCompressionRetriever(BaseRetriever, BaseModel):
    """"""

    llm_chain: LLMChain
    """LLM wrapper to use for compressing documents."""

    base_retriever: BaseRetriever
    """Base Retriever to use for getting relevant documents."""

    get_input: Callable[[str, Document], dict]
    """Callable for constructing the chain input from the query and a Document."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def _compress_docs(self, query: str, docs: List[Document]) -> List[Document]:
        """Compress page content of raw documents."""
        compressed_docs = []
        for doc in docs:
            input = self.get_input(query, doc)
            output = self.llm_chain.predict(**input)
            compressed_docs.append(Document(page_content=output, metadata=doc.metadata))
        return compressed_docs

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            List of relevant documents
        """
        docs = self.base_retriever.get_relevant_documents(query)
        compressed_docs = self._compress_docs(query, docs)
        return compressed_docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            List of relevant documents
        """
        raise NotImplementedError(
            "ContextualCompressionRetriever does not support async."
        )

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: Optional[PromptTemplate] = None,
        get_input: Optional[Callable[[str, Document], str]] = None,
        **kwargs: Any,
    ) -> "ContextualCompressionRetriever":
        """Initialize from LLM."""
        _prompt = prompt if prompt is not None else _get_default_chain_prompt()
        _get_input = get_input if get_input is not None else default_get_input
        llm_chain = LLMChain(llm=llm, prompt=_prompt)
        return cls(llm_chain=llm_chain, get_input=_get_input, **kwargs)
