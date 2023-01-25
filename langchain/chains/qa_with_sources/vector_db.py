"""Question-answering with sources over a vector database."""

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.qa_with_sources.base import BaseQAWithSourcesChain
from langchain.docstore.document import Document
from langchain.vectorstores.base import VectorStore


class VectorDBQAWithSourcesChain(BaseQAWithSourcesChain, BaseModel):
    """Question-answering with sources over a vector database."""

    vectorstore: VectorStore
    """Vector Database to connect to."""
    k: int = 4
    """Number of results to return from store"""
    reduce_k_below_max_tokens: bool = False
    """Reduce the number of results to return from store based on tokens limit"""
    max_tokens_limit: int = 3375
    """Restrict the docs to return from store based on tokens,
    enforced only for StuffDocumentChain and if reduce_k_below_max_tokens is to true"""
    search_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Extra search args."""

    def _page_content(self, doc: Document) -> str:
        return doc.page_content

    def _reduce_tokens_below_limit(self, docs: List[Document]) -> List[Document]:
        if isinstance(self.combine_documents_chain, StuffDocumentsChain):
            tokens = self.combine_documents_chain.llm_chain.llm.get_num_tokens(
                "".join(map(self._page_content, docs))
            )
            return (
                docs
                if (tokens <= self.max_tokens_limit)
                else self._reduce_tokens_below_limit(docs[:-1])
            )
        else:
            return docs

    def _get_docs(self, inputs: Dict[str, Any]) -> List[Document]:
        question = inputs[self.question_key]
        docs = self.vectorstore.similarity_search(
            question, k=self.k, **self.search_kwargs
        )
        return self._reduce_tokens_below_limit(docs)
