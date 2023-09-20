from typing import List

from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.pydantic_v1 import Field
from langchain.schema import BaseRetriever, BaseStore, Document
from langchain.schema.runnable import RunnableLambda
from langchain.vectorstores import VectorStore


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

    def _similarity_search(self, query: str) -> List[Document]:
        """Search for similar documents to a query.
        Args:
            args: A dictionary with the following keys:
                query: The query to search for
                run_manager: The callbacks handler to use
        Returns:
            List of relevant documents
        """
        return self.vectorstore.similarity_search(query, **self.search_kwargs)

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
        sub_docs = RunnableLambda(self._similarity_search).invoke(
            query, {"callbacks": run_manager.get_child()}
        )
        # We do this to maintain the order of the ids that are returned
        ids = []
        for d in sub_docs:
            if d.metadata[self.id_key] not in ids:
                ids.append(d.metadata[self.id_key])
        docs = self.docstore.mget(ids)
        return [d for d in docs if d is not None]
