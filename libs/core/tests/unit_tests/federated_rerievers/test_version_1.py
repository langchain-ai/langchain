"""Design pattern with .indexer attached as an attribute to a retriever."""

from pydantic.v1 import Field
from typing import List, Sequence, Any, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.indexing import UpsertResponse
from langchain_core.indexing.base import DocumentIndex, Content, DeleteResponse
from langchain_core.retrievers import BaseRetriever


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

    def get(self, ids: Sequence[str], **kwargs: Any) -> List[Content]:
        return [self.documents[id] for id in ids]


class InMemoryCatRetriever(BaseRetriever):
    indexer: InMemoryDocIndexer = Field(default_factory=InMemoryDocIndexer)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # ↓↓ Leaks knowledge
        docs = self.indexer.documents  # <-- retriever needs knowledge of the indexer
        # ^^
        good_docs = [doc for doc in docs if "cat" in doc.text.lower()]
        return good_docs


class MultiIndexIndexer(DocumentIndex):
    def __init__(self, indexers: List[DocumentIndex]):
        self.indexers = indexers

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


class FederatedRetriever(BaseRetriever):
    """A simple retriever that returns the first document in the index"""

    retrievers: List[BaseRetriever]
    indexer: DocumentIndex

    def __init__(self, retrievers: List[BaseRetriever]):
        """
        Args:
            retrievers: A list of retrievers to use
            indexer: The indexer to use
        """
        indexer = MultiIndexIndexer([retriever.indexer for retriever in retrievers])
        super().__init__(retrievers=retrievers, indexer=indexer)

    def get_indexer(self) -> DocumentIndex:
        return self.indexer

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        docs = []
        # whatever logic
        for retriever in self.retrievers:
            docs.extend(
                retriever.get_relevant_documents(query, run_manager=run_manager)
            )
        return docs
