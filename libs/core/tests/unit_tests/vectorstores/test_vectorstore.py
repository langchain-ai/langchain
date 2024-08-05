from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Sequence, Union

from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.indexing import UpsertResponse
from langchain_core.vectorstores import VectorStore


def test_custom_upsert_type() -> None:
    """Test that we can override the signature of the upsert method
    of the VectorStore class without creating typing issues by violating
    the Liskov Substitution Principle.
    """

    class ByVector(TypedDict):
        document: Document
        vector: List[float]

    class CustomVectorStore(VectorStore):
        def upsert(
            # This unit test verifies that the signature of the upsert method
            # specifically the items parameter can be overridden without
            # violating the Liskov Substitution Principle (and getting
            # typing errors).
            self,
            items: Union[Sequence[Document], Sequence[ByVector]],
            /,
            **kwargs: Any,
        ) -> UpsertResponse:
            raise NotImplementedError()


class CustomSyncVectorStore(VectorStore):
    """A vectorstore that only implements the synchronous methods."""

    def __init__(self) -> None:
        self.store: Dict[str, Document] = {}

    def upsert(
        self,
        items: Sequence[Document],
        /,
        **kwargs: Any,
    ) -> UpsertResponse:
        ids = []
        for item in items:
            if item.id is None:
                new_item = item.copy()
                id_: str = str(uuid.uuid4())
                new_item.id = id_
            else:
                id_ = item.id
                new_item = item

            self.store[id_] = new_item
            ids.append(id_)

        return {
            "succeeded": ids,
            "failed": [],
        }

    def get_by_ids(self, ids: Sequence[str], /) -> List[Document]:
        return [self.store[id] for id in ids if id in self.store]

    def from_texts(  # type: ignore
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> CustomSyncVectorStore:
        vectorstore = CustomSyncVectorStore()
        vectorstore.add_texts(texts, metadatas=metadatas, **kwargs)
        return vectorstore

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        raise NotImplementedError()


def test_implement_upsert() -> None:
    """Test that we can implement the upsert method of the CustomVectorStore
    class without violating the Liskov Substitution Principle.
    """

    store = CustomSyncVectorStore()

    # Check upsert with id
    assert store.upsert([Document(id="1", page_content="hello")]) == {
        "succeeded": ["1"],
        "failed": [],
    }

    assert store.get_by_ids(["1"]) == [Document(id="1", page_content="hello")]

    # Check upsert without id
    response = store.upsert([Document(page_content="world")])
    assert len(response["succeeded"]) == 1
    id_ = response["succeeded"][0]
    assert id_ is not None
    assert store.get_by_ids([id_]) == [Document(id=id_, page_content="world")]

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

    # Check that add_documents works
    assert store.add_documents([Document(id="5", page_content="baz")]) == ["5"]

    # Test add documents with id specified in both document and ids
    original_document = Document(id="7", page_content="baz")
    assert store.add_documents([original_document], ids=["6"]) == ["6"]
    assert original_document.id == "7"  # original document should not be modified
    assert store.get_by_ids(["6"]) == [Document(id="6", page_content="baz")]


async def test_aupsert_delegation_to_upsert() -> None:
    """Test delegation to the synchronous upsert method in async execution
    if async methods are not implemented.
    """
    store = CustomSyncVectorStore()

    # Check upsert with id
    assert await store.aupsert([Document(id="1", page_content="hello")]) == {
        "succeeded": ["1"],
        "failed": [],
    }

    assert await store.aget_by_ids(["1"]) == [Document(id="1", page_content="hello")]

    # Check upsert without id
    response = await store.aupsert([Document(page_content="world")])
    assert len(response["succeeded"]) == 1
    id_ = response["succeeded"][0]
    assert id_ is not None
    assert await store.aget_by_ids([id_]) == [Document(id=id_, page_content="world")]

    # Check that default implementation of add_texts works
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

    # Check that add_documents works
    assert await store.aadd_documents([Document(id="5", page_content="baz")]) == ["5"]

    # Test add documents with id specified in both document and ids
    original_document = Document(id="7", page_content="baz")
    assert await store.aadd_documents([original_document], ids=["6"]) == ["6"]
    assert original_document.id == "7"  # original document should not be modified
    assert await store.aget_by_ids(["6"]) == [Document(id="6", page_content="baz")]
