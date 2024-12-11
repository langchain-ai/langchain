"""Set of tests that complement the standard tests for vectorstore.

These tests verify that the base abstraction does appropriate delegation to
the relevant methods.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from typing import Any, Optional

import pytest

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings, FakeEmbeddings
from langchain_core.vectorstores import VectorStore


class CustomAddTextsVectorstore(VectorStore):
    """A vectorstore that only implements add texts."""

    def __init__(self) -> None:
        self.store: dict[str, Document] = {}

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        if not isinstance(texts, list):
            texts = list(texts)
        ids_iter = iter(ids or [])

        ids_ = []

        metadatas_ = metadatas or [{} for _ in texts]

        for text, metadata in zip(texts, metadatas_ or []):
            next_id = next(ids_iter, None)
            id_ = next_id or str(uuid.uuid4())
            self.store[id_] = Document(page_content=text, metadata=metadata, id=id_)
            ids_.append(id_)
        return ids_

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        return [self.store[id] for id in ids if id in self.store]

    @classmethod
    def from_texts(  # type: ignore
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: Optional[list[dict]] = None,
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

    def add_documents(
        self,
        documents: list[Document],
        *,
        ids: Optional[list[str]] = None,
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
        return [self.store[id] for id in ids if id in self.store]

    @classmethod
    def from_texts(  # type: ignore
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: Optional[list[dict]] = None,
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
    """Test that we can implement the upsert method of the CustomVectorStore
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
