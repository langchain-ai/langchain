from enum import Enum
from typing import List, Optional, Any

from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field, validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.stores import BaseStore
from langchain_core.vectorstores import VectorStore

from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.storage._lc_store import create_kv_docstore


class SearchType(str, Enum):
    """Enumerator of the types of search to perform."""

    similarity = "similarity"
    """Similarity search."""
    mmr = "mmr"
    """Maximal Marginal Relevance reranking of similarity search."""


class MultiVectorRetriever(BaseRetriever):
    """Retrieve from a set of multiple embeddings for the same document."""

    vectorstore: VectorStore
    """The underlying vectorstore to use to store small chunks
    and their embedding vectors"""
    base_store: Optional[BaseStore[str, bytes]]
    """The lower-level backing storage layer for the parent documents"""
    docstore: Optional[BaseStore[str, Document]]
    """The storage interface for the parent documents"""
    id_key: str = "doc_id"
    search_kwargs: dict = Field(default_factory=dict)
    """Keyword arguments to pass to the search function."""
    search_type: SearchType = SearchType.similarity
    """Type of search to perform (similarity / mmr)"""

    @validator("docstore", pre=True, always=True)
    def shim_docstore(
        cls, docstore: Optional[BaseStore[str, Document]], values: any
    ) -> BaseStore[str, Document]:
        base_store = values.get("base_store")
        if base_store is not None:
            docstore = create_kv_docstore(base_store)
        elif docstore is None:
            raise Exception("You must pass a `base_store` parameter.")
        return docstore

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to a query.
        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents
        """
        if self.search_type == SearchType.mmr:
            sub_docs = self.vectorstore.max_marginal_relevance_search(
                query, **self.search_kwargs
            )
        else:
            sub_docs = self.vectorstore.similarity_search(query, **self.search_kwargs)

        # We do this to maintain the order of the ids that are returned
        ids = []
        for d in sub_docs:
            if d.metadata[self.id_key] not in ids:
                ids.append(d.metadata[self.id_key])
        docs = self.docstore.mget(ids)
        return [d for d in docs if d is not None]
