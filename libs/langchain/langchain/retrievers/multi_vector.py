from enum import Enum
from typing import List, Optional

from langchain_core.documents import Document
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
    docstore: BaseStore[str, Document]
    """The storage layer for the parent documents"""
    id_key: str
    search_kwargs: dict
    """Keyword arguments to pass to the search function."""
    search_type: SearchType
    """Type of search to perform (similarity / mmr)"""

    def __init__(
        self,
        *,
        vectorstore: VectorStore,
        docstore: Optional[BaseStore[str, Document]] = None,
        base_store: Optional[BaseStore[str, bytes]] = None,
        id_key: str = "doc_id",
        search_kwargs: Optional[dict] = None,
        search_type: SearchType = SearchType.similarity,
    ):
        if base_store is not None:
            docstore = create_kv_docstore(base_store)
        elif docstore is None:
            raise Exception("You must pass a `base_store` parameter.")

        super().__init__(
            vectorstore=vectorstore,
            docstore=docstore,
            id_key=id_key,
            search_kwargs=search_kwargs if search_kwargs is not None else {},
            search_type=search_type,
        )

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
