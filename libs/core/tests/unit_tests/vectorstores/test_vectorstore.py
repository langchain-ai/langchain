"""Set of tests that complement the standard tests for vectorstore.

These tests verify that the base abstraction does appropriate delegation to
the relevant methods.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import pytest
from typing_extensions import override

from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.embeddings import Embeddings, FakeEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores.base import ContextualCompressionRetriever

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun, Callbacks
    from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun


class CustomAddTextsVectorstore(VectorStore):
    """A vectorstore that only implements add texts."""

    def __init__(self) -> None:
        self.store: dict[str, Document] = {}

    @override
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        if not isinstance(texts, list):
            texts = list(texts)
        ids_iter = iter(ids or [])

        ids_ = []

        metadatas_ = metadatas or [{} for _ in texts]

        for text, metadata in zip(texts, metadatas_ or [], strict=False):
            next_id = next(ids_iter, None)
            id_ = next_id or str(uuid.uuid4())
            self.store[id_] = Document(page_content=text, metadata=metadata, id=id_)
            ids_.append(id_)
        return ids_

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        return [self.store[id_] for id_ in ids if id_ in self.store]

    @classmethod
    @override
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        **kwargs: Any,
    ) -> CustomAddTextsVectorstore:
        vectorstore = CustomAddTextsVectorstore()
        vectorstore.add_texts(texts, metadatas=metadatas, **kwargs)
        return vectorstore

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        raise NotImplementedError


class CustomAddDocumentsVectorstore(VectorStore):
    """A vectorstore that only implements add documents."""

    def __init__(self) -> None:
        self.store: dict[str, Document] = {}

    @override
    def add_documents(
        self,
        documents: list[Document],
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        ids_ = []
        ids_iter = iter(ids or [])
        for document in documents:
            id_ = next(ids_iter) if ids else document.id or str(uuid.uuid4())
            self.store[id_] = Document(
                id=id_, page_content=document.page_content, metadata=document.metadata
            )
            ids_.append(id_)
        return ids_

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        return [self.store[id_] for id_ in ids if id_ in self.store]

    @classmethod
    @override
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        **kwargs: Any,
    ) -> CustomAddDocumentsVectorstore:
        vectorstore = CustomAddDocumentsVectorstore()
        vectorstore.add_texts(texts, metadatas=metadatas, **kwargs)
        return vectorstore

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        raise NotImplementedError


@pytest.mark.parametrize(
    "vs_class", [CustomAddTextsVectorstore, CustomAddDocumentsVectorstore]
)
def test_default_add_documents(vs_class: type[VectorStore]) -> None:
    """Test default implementation of add_documents.

    Test that we can implement the upsert method of the CustomVectorStore
    class without violating the Liskov Substitution Principle.
    """
    store = vs_class()

    # Check upsert with id
    assert store.add_documents([Document(id="1", page_content="hello")]) == ["1"]

    assert store.get_by_ids(["1"]) == [Document(id="1", page_content="hello")]

    # Check upsert without id
    ids = store.add_documents([Document(page_content="world")])
    assert len(ids) == 1
    assert store.get_by_ids(ids) == [Document(id=ids[0], page_content="world")]

    # Check that add_documents works
    assert store.add_documents([Document(id="5", page_content="baz")]) == ["5"]

    # Test add documents with id specified in both document and ids
    original_document = Document(id="7", page_content="baz")
    assert store.add_documents([original_document], ids=["6"]) == ["6"]
    assert original_document.id == "7"  # original document should not be modified
    assert store.get_by_ids(["6"]) == [Document(id="6", page_content="baz")]


@pytest.mark.parametrize(
    "vs_class", [CustomAddTextsVectorstore, CustomAddDocumentsVectorstore]
)
def test_default_add_texts(vs_class: type[VectorStore]) -> None:
    store = vs_class()
    # Check that default implementation of add_texts works
    assert store.add_texts(["hello", "world"], ids=["3", "4"]) == ["3", "4"]

    assert store.get_by_ids(["3", "4"]) == [
        Document(id="3", page_content="hello"),
        Document(id="4", page_content="world"),
    ]

    # Add texts without ids
    ids_ = store.add_texts(["foo", "bar"])
    assert len(ids_) == 2
    assert store.get_by_ids(ids_) == [
        Document(id=ids_[0], page_content="foo"),
        Document(id=ids_[1], page_content="bar"),
    ]

    # Add texts with metadatas
    ids_2 = store.add_texts(["foo", "bar"], metadatas=[{"foo": "bar"}] * 2)
    assert len(ids_2) == 2
    assert store.get_by_ids(ids_2) == [
        Document(id=ids_2[0], page_content="foo", metadata={"foo": "bar"}),
        Document(id=ids_2[1], page_content="bar", metadata={"foo": "bar"}),
    ]


@pytest.mark.parametrize(
    "vs_class", [CustomAddTextsVectorstore, CustomAddDocumentsVectorstore]
)
async def test_default_aadd_documents(vs_class: type[VectorStore]) -> None:
    """Test delegation to the synchronous method."""
    store = vs_class()

    # Check upsert with id
    assert await store.aadd_documents([Document(id="1", page_content="hello")]) == ["1"]

    assert await store.aget_by_ids(["1"]) == [Document(id="1", page_content="hello")]

    # Check upsert without id
    ids = await store.aadd_documents([Document(page_content="world")])
    assert len(ids) == 1
    assert await store.aget_by_ids(ids) == [Document(id=ids[0], page_content="world")]

    # Check that add_documents works
    assert await store.aadd_documents([Document(id="5", page_content="baz")]) == ["5"]

    # Test add documents with id specified in both document and ids
    original_document = Document(id="7", page_content="baz")
    assert await store.aadd_documents([original_document], ids=["6"]) == ["6"]
    assert original_document.id == "7"  # original document should not be modified
    assert await store.aget_by_ids(["6"]) == [Document(id="6", page_content="baz")]


@pytest.mark.parametrize(
    "vs_class", [CustomAddTextsVectorstore, CustomAddDocumentsVectorstore]
)
async def test_default_aadd_texts(vs_class: type[VectorStore]) -> None:
    """Test delegation to the synchronous method."""
    store = vs_class()
    # Check that default implementation of aadd_texts works
    assert await store.aadd_texts(["hello", "world"], ids=["3", "4"]) == ["3", "4"]

    assert await store.aget_by_ids(["3", "4"]) == [
        Document(id="3", page_content="hello"),
        Document(id="4", page_content="world"),
    ]

    # Add texts without ids
    ids_ = await store.aadd_texts(["foo", "bar"])
    assert len(ids_) == 2
    assert await store.aget_by_ids(ids_) == [
        Document(id=ids_[0], page_content="foo"),
        Document(id=ids_[1], page_content="bar"),
    ]

    # Add texts with metadatas
    ids_2 = await store.aadd_texts(["foo", "bar"], metadatas=[{"foo": "bar"}] * 2)
    assert len(ids_2) == 2
    assert await store.aget_by_ids(ids_2) == [
        Document(id=ids_2[0], page_content="foo", metadata={"foo": "bar"}),
        Document(id=ids_2[1], page_content="bar", metadata={"foo": "bar"}),
    ]


@pytest.mark.parametrize(
    "vs_class", [CustomAddTextsVectorstore, CustomAddDocumentsVectorstore]
)
def test_default_from_documents(vs_class: type[VectorStore]) -> None:
    embeddings = FakeEmbeddings(size=1)
    store = vs_class.from_documents(
        [Document(id="1", page_content="hello", metadata={"foo": "bar"})], embeddings
    )

    assert store.get_by_ids(["1"]) == [
        Document(id="1", page_content="hello", metadata={"foo": "bar"})
    ]

    # from_documents with ids in args
    store = vs_class.from_documents(
        [Document(page_content="hello", metadata={"foo": "bar"})], embeddings, ids=["1"]
    )

    assert store.get_by_ids(["1"]) == [
        Document(id="1", page_content="hello", metadata={"foo": "bar"})
    ]

    # Test from_documents with id specified in both document and ids
    original_document = Document(id="7", page_content="baz")
    store = vs_class.from_documents([original_document], embeddings, ids=["6"])
    assert original_document.id == "7"  # original document should not be modified
    assert store.get_by_ids(["6"]) == [Document(id="6", page_content="baz")]


@pytest.mark.parametrize(
    "vs_class", [CustomAddTextsVectorstore, CustomAddDocumentsVectorstore]
)
async def test_default_afrom_documents(vs_class: type[VectorStore]) -> None:
    embeddings = FakeEmbeddings(size=1)
    store = await vs_class.afrom_documents(
        [Document(id="1", page_content="hello", metadata={"foo": "bar"})], embeddings
    )

    assert await store.aget_by_ids(["1"]) == [
        Document(id="1", page_content="hello", metadata={"foo": "bar"})
    ]

    # from_documents with ids in args
    store = await vs_class.afrom_documents(
        [Document(page_content="hello", metadata={"foo": "bar"})], embeddings, ids=["1"]
    )

    assert await store.aget_by_ids(["1"]) == [
        Document(id="1", page_content="hello", metadata={"foo": "bar"})
    ]

    # Test afrom_documents with id specified in both document and ids
    original_document = Document(id="7", page_content="baz")
    store = await vs_class.afrom_documents([original_document], embeddings, ids=["6"])
    assert original_document.id == "7"  # original document should not be modified
    assert await store.aget_by_ids(["6"]) == [Document(id="6", page_content="baz")]


class FakeRetriever(BaseRetriever):
    """Fake retriever for testing."""

    docs: list[Document] = []
    """Documents to return."""

    @override
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> list[Document]:
        """Return the documents."""
        return self.docs

    @override
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> list[Document]:
        """Async return the documents."""
        return self.docs


class FakeCompressor(BaseDocumentCompressor):
    """Fake compressor for testing."""

    filter_fn: Any | None = None
    """Optional filter function to apply to documents."""

    @override
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        """Compress documents."""
        if self.filter_fn:
            return [doc for doc in documents if self.filter_fn(doc, query)]
        # Default: return first half of documents
        return documents[: len(documents) // 2] if documents else []

    @override
    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        """Async compress documents."""
        if self.filter_fn:
            return [doc for doc in documents if self.filter_fn(doc, query)]
        # Default: return first half of documents
        return documents[: len(documents) // 2] if documents else []


def test_contextual_compression_retriever_sync() -> None:
    """Test ContextualCompressionRetriever synchronous retrieval."""
    # Create base retriever with documents
    docs = [
        Document(page_content="doc1", metadata={"id": 1}),
        Document(page_content="doc2", metadata={"id": 2}),
        Document(page_content="doc3", metadata={"id": 3}),
        Document(page_content="doc4", metadata={"id": 4}),
    ]
    base_retriever = FakeRetriever(docs=docs)

    # Create compressor that returns first half
    compressor = FakeCompressor()

    # Create contextual compression retriever
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    # Test retrieval
    result = retriever.invoke("test query")

    # Should return first 2 documents (first half of 4)
    assert len(result) == 2
    assert result[0].page_content == "doc1"
    assert result[1].page_content == "doc2"


async def test_contextual_compression_retriever_async() -> None:
    """Test ContextualCompressionRetriever asynchronous retrieval."""
    # Create base retriever with documents
    docs = [
        Document(page_content="doc1", metadata={"id": 1}),
        Document(page_content="doc2", metadata={"id": 2}),
        Document(page_content="doc3", metadata={"id": 3}),
        Document(page_content="doc4", metadata={"id": 4}),
    ]
    base_retriever = FakeRetriever(docs=docs)

    # Create compressor that returns first half
    compressor = FakeCompressor()

    # Create contextual compression retriever
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    # Test async retrieval
    result = await retriever.ainvoke("test query")

    # Should return first 2 documents (first half of 4)
    assert len(result) == 2
    assert result[0].page_content == "doc1"
    assert result[1].page_content == "doc2"


def test_contextual_compression_retriever_empty_results() -> None:
    """Test ContextualCompressionRetriever with empty results from base retriever."""
    # Create base retriever with no documents
    base_retriever = FakeRetriever(docs=[])

    # Create compressor
    compressor = FakeCompressor()

    # Create contextual compression retriever
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    # Test retrieval with empty results
    result = retriever.invoke("test query")

    # Should return empty list
    assert result == []


async def test_contextual_compression_retriever_empty_results_async() -> None:
    """Test ContextualCompressionRetriever async with empty results."""
    # Create base retriever with no documents
    base_retriever = FakeRetriever(docs=[])

    # Create compressor
    compressor = FakeCompressor()

    # Create contextual compression retriever
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    # Test async retrieval with empty results
    result = await retriever.ainvoke("test query")

    # Should return empty list
    assert result == []


def test_contextual_compression_retriever_custom_filter() -> None:
    """Test ContextualCompressionRetriever with custom filtering logic."""
    # Create base retriever with documents
    docs = [
        Document(page_content="apple", metadata={"category": "fruit"}),
        Document(page_content="carrot", metadata={"category": "vegetable"}),
        Document(page_content="banana", metadata={"category": "fruit"}),
        Document(page_content="broccoli", metadata={"category": "vegetable"}),
    ]
    base_retriever = FakeRetriever(docs=docs)

    # Create compressor that filters for fruits only
    def filter_fruits(doc: Document, _query: str) -> bool:
        return doc.metadata.get("category") == "fruit"

    compressor = FakeCompressor(filter_fn=filter_fruits)

    # Create contextual compression retriever
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    # Test retrieval
    result = retriever.invoke("test query")

    # Should return only fruit documents
    assert len(result) == 2
    assert result[0].page_content == "apple"
    assert result[1].page_content == "banana"


async def test_contextual_compression_retriever_custom_filter_async() -> None:
    """Test ContextualCompressionRetriever async with custom filtering logic."""
    # Create base retriever with documents
    docs = [
        Document(page_content="apple", metadata={"category": "fruit"}),
        Document(page_content="carrot", metadata={"category": "vegetable"}),
        Document(page_content="banana", metadata={"category": "fruit"}),
        Document(page_content="broccoli", metadata={"category": "vegetable"}),
    ]
    base_retriever = FakeRetriever(docs=docs)

    # Create compressor that filters for fruits only
    def filter_fruits(doc: Document, _query: str) -> bool:
        return doc.metadata.get("category") == "fruit"

    compressor = FakeCompressor(filter_fn=filter_fruits)

    # Create contextual compression retriever
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    # Test async retrieval
    result = await retriever.ainvoke("test query")

    # Should return only fruit documents
    assert len(result) == 2
    assert result[0].page_content == "apple"
    assert result[1].page_content == "banana"


def test_contextual_compression_retriever_all_filtered_out() -> None:
    """Test ContextualCompressionRetriever when all documents are filtered out."""
    # Create base retriever with documents
    docs = [
        Document(page_content="doc1", metadata={"score": 0.1}),
        Document(page_content="doc2", metadata={"score": 0.2}),
    ]
    base_retriever = FakeRetriever(docs=docs)

    # Create compressor that filters out all documents
    def filter_high_scores(doc: Document, _query: str) -> bool:
        return doc.metadata.get("score", 0.0) > 0.9

    compressor = FakeCompressor(filter_fn=filter_high_scores)

    # Create contextual compression retriever
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    # Test retrieval
    result = retriever.invoke("test query")

    # Should return empty list
    assert result == []


def test_contextual_compression_retriever_callbacks() -> None:
    """Test that ContextualCompressionRetriever properly invokes compressor."""
    # Create base retriever with documents
    docs = [
        Document(page_content="doc1"),
        Document(page_content="doc2"),
        Document(page_content="doc3"),
        Document(page_content="doc4"),
    ]
    base_retriever = FakeRetriever(docs=docs)

    # Track if compress_documents was called
    compress_called = []

    class TrackingCompressor(BaseDocumentCompressor):
        """Compressor that tracks method calls."""

        def compress_documents(
            self,
            documents: Sequence[Document],
            query: str,
            callbacks: Callbacks | None = None,
        ) -> Sequence[Document]:
            """Compress documents and track the call."""
            compress_called.append(
                {"documents": documents, "query": query, "callbacks": callbacks}
            )
            return documents[:1]  # Return first document only

    compressor = TrackingCompressor()

    # Create contextual compression retriever
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    # Test retrieval with callbacks
    result = retriever.invoke("test query")

    # Verify compress_documents was called
    assert len(compress_called) == 1
    # Verify query was passed
    assert compress_called[0]["query"] == "test query"
    # Verify callbacks were passed (not None in this context)
    assert compress_called[0]["callbacks"] is not None
    # Verify result is compressed (1 doc instead of 4)
    assert len(result) == 1
    assert result[0].page_content == "doc1"
