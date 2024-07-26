"""Design pattern where the Indexer implements a .search() method.

We can either subclass from Retriever and document index is a retriever that
invokes the search() method correctly.

InMemoryDocIndexer.invoke('meow')

Or we can create a retriever from it using a factory method -- this amounts
to chopping off everything but the search() method and standardizing the inputs
and outputs into that method.

InMemoryDocIndexer.create_retriever().invoke('meow')
"""
from typing import List, Sequence, Any, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.indexing import UpsertResponse
from langchain_core.indexing.base import (
    DocumentIndex,
    Content,
    DeleteResponse,
    QueryResponse,
    Hit,
)


class InMemoryDocIndexer(DocumentIndex):
    def __init__(self):
        self.documents = {}

    def upsert(self, items: Sequence[Content], /, **kwargs: Any) -> UpsertResponse:
        for item in items:
            self.documents[item.id] = item
        return UpsertResponse(success=True)

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> DeleteResponse:
        failed_ids = []
        for id in ids:
            try:
                del self.documents[id]
            except KeyError:
                failed_ids.append(id)
        return sorted(set(failed_ids))

    def get_by_id(self, id: str, **kwargs: Any) -> Optional[Content]:
        return self.documents.get(id)

    def search(self, query: str, **kwargs: Any) -> QueryResponse[Hit]:
        good_docs = [doc for doc in docs if "cat" in doc.text.lower()]
        return QueryResponse(
            hits=[Hit(score=1.0, **doc) for doc in good_docs]
        )

class FederatedIndex(DocumentIndex):
    """A simple retriever that returns the first document in the index"""

    indexes: List[DocumentIndex]

    def upsert(self, items: Sequence[Content], /, **kwargs: Any) -> UpsertResponse:
        for index in self.indexers:
            index.upsert(items, **kwargs)
        return {}  # Update properly in actual implementation

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> DeleteResponse:
        failed_ids = []
        # Thread pool
        for index in self.indexers:
            try:
                index.delete(ids, **kwargs)
            except Exception as e:
                failed_ids.extend(ids)
        return sorted(set(failed_ids))

    def get(
        self, ids: Sequence[str], *, index_id: int = 0, **kwargs
    ) -> Optional[Content]:
        indexer = self.indexers[index_id]
        return indexer.get(ids, **kwargs)

    def search(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        docs = []
        # whatever logic
        for retriever in self.retrievers:
            docs.extend(
                retriever.get_relevant_documents(query, run_manager=run_manager)
            )
        return docs
