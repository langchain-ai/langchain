"""Set of tests that complement the standard tests for vectorstore.

These tests verify that the base abstraction does appropriate delegation to
the relevant methods.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import pytest
from typing_extensions import override

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings, FakeEmbeddings
from langchain_core.vectorstores import VectorStore

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


class CustomAddTextsVectorstore(VectorStore):
    """A VectorStore that only implements add texts."""

    def __init__(self) -> None:
        self.store: dict[str, Document] = {}

    @override
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None = None,
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
        metadatas: list[dict[str, Any]] | None = None,
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
    """A VectorStore that only implements add documents."""

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
        metadatas: list[dict[str, Any]] | None = None,
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

    # from_documents with IDs in args
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

    # from_documents with IDs in args
    store = await vs_class.afrom_documents(
        [Document(page_content="hello", metadata={"foo": "bar"})], embeddings, ids=["1"]
    )

    assert await store.aget_by_ids(["1"]) == [
        Document(id="1", page_content="hello", metadata={"foo": "bar"})
    ]

    # Test afrom_documents with id specified in both document and IDs
    original_document = Document(id="7", page_content="baz")
    store = await vs_class.afrom_documents([original_document], embeddings, ids=["6"])
    assert original_document.id == "7"  # original document should not be modified
    assert await store.aget_by_ids(["6"]) == [Document(id="6", page_content="baz")]


# ---------------------------------------------------------------------------
# Regression tests for https://github.com/langchain-ai/langchain/issues/32283
#
# When a VectorStore subclass overrides add_documents / aadd_documents with
# `ids` as an *explicit* keyword-only parameter (like QdrantVectorStore does),
# calling add_texts / aadd_texts with `ids` must NOT forward ids twice —
# once through the explicit parameter AND once through **kwargs — otherwise
# Python raises "TypeError: got multiple values for keyword argument 'ids'".
# ---------------------------------------------------------------------------


class ExplicitIdsAddDocumentsVectorstore(VectorStore):
    """A VectorStore whose add_documents declares `ids` as an explicit kwarg.

    This mirrors the signature used by QdrantVectorStore and similar
    third-party integrations that triggered the 'multiple values' bug.
    """

    def __init__(self) -> None:
        self.store: dict[str, Document] = {}
        # Track how many times ids was received so we can assert no duplication.
        self.add_documents_calls: list[dict] = []

    @override
    def add_documents(
        self,
        documents: list[Document],
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Override with ids as an explicit keyword-only parameter."""
        # ids must NOT also appear inside kwargs — that is the regression.
        assert "ids" not in kwargs, (
            "ids was passed both explicitly and via **kwargs — "
            "regression of github.com/langchain-ai/langchain/issues/32283"
        )
        self.add_documents_calls.append({"ids": ids, "kwargs": kwargs})
        result_ids: list[str] = []
        import uuid

        ids_iter = iter(ids or [])
        for doc in documents:
            id_ = next(ids_iter, None) or doc.id or str(uuid.uuid4())
            self.store[id_] = Document(
                id=id_, page_content=doc.page_content, metadata=doc.metadata
            )
            result_ids.append(id_)
        return result_ids

    @override
    async def aadd_documents(
        self,
        documents: list[Document],
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Async override with ids as an explicit keyword-only parameter."""
        # ids must NOT also appear inside kwargs — that is the regression.
        assert "ids" not in kwargs, (
            "ids was passed both explicitly and via **kwargs — "
            "regression of github.com/langchain-ai/langchain/issues/32283"
        )
        return self.add_documents(documents, ids=ids, **kwargs)

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        return [self.store[id_] for id_ in ids if id_ in self.store]

    @classmethod
    @override
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ExplicitIdsAddDocumentsVectorstore:
        store = ExplicitIdsAddDocumentsVectorstore()
        store.add_texts(texts, metadatas=metadatas, **kwargs)
        return store

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        raise NotImplementedError


def test_add_texts_with_explicit_ids_no_duplicate_kwarg() -> None:
    """Regression test for #32283 (sync path).

    add_texts(ids=...) must not pass `ids` twice to add_documents when the
    subclass declares `ids` as an explicit keyword-only parameter.
    """
    store = ExplicitIdsAddDocumentsVectorstore()
    # This must not raise TypeError: got multiple values for keyword argument 'ids'
    returned_ids = store.add_texts(["hello", "world"], ids=["id-1", "id-2"])
    assert returned_ids == ["id-1", "id-2"]
    assert store.get_by_ids(["id-1", "id-2"]) == [
        Document(id="id-1", page_content="hello"),
        Document(id="id-2", page_content="world"),
    ]


def test_add_texts_without_ids_no_duplicate_kwarg() -> None:
    """Regression test for #32283 (sync path, no ids).

    add_texts() without ids must not accidentally inject ids into kwargs.
    """
    store = ExplicitIdsAddDocumentsVectorstore()
    returned_ids = store.add_texts(["foo", "bar"])
    assert len(returned_ids) == 2
    # Verify each call received ids=None (or a list), never ids inside kwargs
    for call in store.add_documents_calls:
        assert "ids" not in call["kwargs"]


async def test_aadd_texts_with_explicit_ids_no_duplicate_kwarg() -> None:
    """Regression test for #32283 (async path).

    aadd_texts(ids=...) must not pass `ids` twice to aadd_documents when the
    subclass declares `ids` as an explicit keyword-only parameter.
    """
    store = ExplicitIdsAddDocumentsVectorstore()
    # This must not raise TypeError: got multiple values for keyword argument 'ids'
    returned_ids = await store.aadd_texts(["hello", "world"], ids=["id-1", "id-2"])
    assert returned_ids == ["id-1", "id-2"]
    assert await store.aget_by_ids(["id-1", "id-2"]) == [
        Document(id="id-1", page_content="hello"),
        Document(id="id-2", page_content="world"),
    ]


async def test_aadd_texts_without_ids_no_duplicate_kwarg() -> None:
    """Regression test for #32283 (async path, no ids).

    aadd_texts() without ids must not accidentally inject ids into kwargs.
    """
    store = ExplicitIdsAddDocumentsVectorstore()
    returned_ids = await store.aadd_texts(["foo", "bar"])
    assert len(returned_ids) == 2


async def test_aadd_documents_with_explicit_ids_no_duplicate_kwarg() -> None:
    """Regression test for #32283 (async aadd_documents delegation path).

    aadd_documents(ids=...) must forward ids exactly once to aadd_texts when
    the subclass overrides aadd_texts with an explicit `ids` parameter.
    """
    store = ExplicitIdsAddDocumentsVectorstore()
    returned_ids = await store.aadd_documents(
        [Document(page_content="hello"), Document(page_content="world")],
        ids=["id-a", "id-b"],
    )
    assert returned_ids == ["id-a", "id-b"]
    assert await store.aget_by_ids(["id-a", "id-b"]) == [
        Document(id="id-a", page_content="hello"),
        Document(id="id-b", page_content="world"),
    ]
