import datetime
from copy import deepcopy
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from pydantic import ConfigDict, Field
from typing_extensions import override


def _get_hours_passed(time: datetime.datetime, ref_time: datetime.datetime) -> float:
    """Get the hours passed between two datetimes."""
    return (time - ref_time).total_seconds() / 3600


class TimeWeightedVectorStoreRetriever(BaseRetriever):
    """Time Weighted Vector Store Retriever.

    Retriever that combines embedding similarity with recency in retrieving values.
    """

    vectorstore: VectorStore
    """The `VectorStore` to store documents and determine salience."""

    search_kwargs: dict = Field(default_factory=lambda: {"k": 100})
    """Keyword arguments to pass to the `VectorStore` similarity search."""

    # TODO: abstract as a queue
    memory_stream: list[Document] = Field(default_factory=list)
    """The memory_stream of documents to search through."""

    decay_rate: float = Field(default=0.01)
    """The exponential decay factor used as `(1.0-decay_rate)**(hrs_passed)`."""

    k: int = 4
    """The maximum number of documents to retrieve in a given call."""

    other_score_keys: list[str] = []
    """Other keys in the metadata to factor into the score, e.g. 'importance'."""

    default_salience: float | None = None
    """The salience to assign memories not retrieved from the vector store.

    None assigns no salience to documents not fetched from the vector store.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def _document_get_date(self, field: str, document: Document) -> datetime.datetime:
        """Return the value of the date field of a document."""
        if field in document.metadata:
            if isinstance(document.metadata[field], float):
                return datetime.datetime.fromtimestamp(document.metadata[field])
            return document.metadata[field]
        return datetime.datetime.now()

    def _get_combined_score(
        self,
        document: Document,
        vector_relevance: float | None,
        current_time: datetime.datetime,
    ) -> float:
        """Return the combined score for a document."""
        hours_passed = _get_hours_passed(
            current_time,
            self._document_get_date("last_accessed_at", document),
        )
        score = (1.0 - self.decay_rate) ** hours_passed
        for key in self.other_score_keys:
            if key in document.metadata:
                score += document.metadata[key]
        if vector_relevance is not None:
            score += vector_relevance
        return score

    def get_salient_docs(self, query: str) -> dict[int, tuple[Document, float]]:
        """Return documents that are salient to the query."""
        docs_and_scores: list[tuple[Document, float]]
        docs_and_scores = self.vectorstore.similarity_search_with_relevance_scores(
            query,
            **self.search_kwargs,
        )
        results = {}
        for fetched_doc, relevance in docs_and_scores:
            if "buffer_idx" in fetched_doc.metadata:
                buffer_idx = fetched_doc.metadata["buffer_idx"]
                doc = self.memory_stream[buffer_idx]
                results[buffer_idx] = (doc, relevance)
        return results

    async def aget_salient_docs(self, query: str) -> dict[int, tuple[Document, float]]:
        """Return documents that are salient to the query."""
        docs_and_scores: list[tuple[Document, float]]
        docs_and_scores = (
            await self.vectorstore.asimilarity_search_with_relevance_scores(
                query,
                **self.search_kwargs,
            )
        )
        results = {}
        for fetched_doc, relevance in docs_and_scores:
            if "buffer_idx" in fetched_doc.metadata:
                buffer_idx = fetched_doc.metadata["buffer_idx"]
                doc = self.memory_stream[buffer_idx]
                results[buffer_idx] = (doc, relevance)
        return results

    def _get_rescored_docs(
        self,
        docs_and_scores: dict[Any, tuple[Document, float | None]],
    ) -> list[Document]:
        current_time = datetime.datetime.now()
        rescored_docs = [
            (doc, self._get_combined_score(doc, relevance, current_time))
            for doc, relevance in docs_and_scores.values()
        ]
        rescored_docs.sort(key=lambda x: x[1], reverse=True)
        result = []
        # Ensure frequently accessed memories aren't forgotten
        for doc, _ in rescored_docs[: self.k]:
            # TODO: Update vector store doc once `update` method is exposed.
            buffered_doc = self.memory_stream[doc.metadata["buffer_idx"]]
            buffered_doc.metadata["last_accessed_at"] = current_time
            result.append(buffered_doc)
        return result

    @override
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        docs_and_scores = {
            doc.metadata["buffer_idx"]: (doc, self.default_salience)
            for doc in self.memory_stream[-self.k :]
        }
        # If a doc is considered salient, update the salience score
        docs_and_scores.update(self.get_salient_docs(query))
        return self._get_rescored_docs(docs_and_scores)

    @override
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        docs_and_scores = {
            doc.metadata["buffer_idx"]: (doc, self.default_salience)
            for doc in self.memory_stream[-self.k :]
        }
        # If a doc is considered salient, update the salience score
        docs_and_scores.update(await self.aget_salient_docs(query))
        return self._get_rescored_docs(docs_and_scores)

    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        """Add documents to vectorstore."""
        current_time = kwargs.get("current_time")
        if current_time is None:
            current_time = datetime.datetime.now()
        # Avoid mutating input documents
        dup_docs = [deepcopy(d) for d in documents]
        for i, doc in enumerate(dup_docs):
            if "last_accessed_at" not in doc.metadata:
                doc.metadata["last_accessed_at"] = current_time
            if "created_at" not in doc.metadata:
                doc.metadata["created_at"] = current_time
            doc.metadata["buffer_idx"] = len(self.memory_stream) + i
        self.memory_stream.extend(dup_docs)
        return self.vectorstore.add_documents(dup_docs, **kwargs)

    async def aadd_documents(
        self,
        documents: list[Document],
        **kwargs: Any,
    ) -> list[str]:
        """Add documents to vectorstore."""
        current_time = kwargs.get("current_time")
        if current_time is None:
            current_time = datetime.datetime.now()
        # Avoid mutating input documents
        dup_docs = [deepcopy(d) for d in documents]
        for i, doc in enumerate(dup_docs):
            if "last_accessed_at" not in doc.metadata:
                doc.metadata["last_accessed_at"] = current_time
            if "created_at" not in doc.metadata:
                doc.metadata["created_at"] = current_time
            doc.metadata["buffer_idx"] = len(self.memory_stream) + i
        self.memory_stream.extend(dup_docs)
        return await self.vectorstore.aadd_documents(dup_docs, **kwargs)
