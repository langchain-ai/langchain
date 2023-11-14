from typing import List, Optional

from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.pydantic_v1 import Field
from langchain.schema import BaseRetriever, BaseStore, Document
from langchain.schema.vectorstore import VectorStore


class MultiVectorRetriever(BaseRetriever):
    """Retrieve from a set of multiple embeddings for the same document."""

    vectorstore: VectorStore
    """The underlying vectorstore to use to store small chunks
    and their embedding vectors"""
    docstore: BaseStore[str, Document]
    """The storage layer for the parent documents"""
    id_key: str = "doc_id"
    search_kwargs: dict = Field(default_factory=dict)
    """Keyword arguments to pass to the search function."""

    docstore_k: Optional[int] = None
    """The k on the underlying vectorstore can be set to a high number, 
    to get a lot of matches especially if docs in the docstore are large.
    In this case, the docstore_k parameter can be set to a lower number 
    to reduce context passed to the LLM."""

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
        sub_docs = self.vectorstore.similarity_search(query, **self.search_kwargs)
        # We do this to maintain the order of the ids that are returned
        ids = []
        for d in sub_docs:
            if d.metadata[self.id_key] not in ids:
                ids.append(d.metadata[self.id_key])
        docs = self.docstore.mget(ids)
        docs = [d for d in docs if d is not None]
        if self.docstore_k is not None:
            docs = docs[: self.docstore_k]

        return docs
