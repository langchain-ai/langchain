from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


class AsyncTextVectorStore(VectorStore):
    def __init__(self) -> None:
        self.received_texts: list[str] | None = None
        self.received_metadatas: list[dict[str, Any]] | None = None
        self.received_ids: list[str] | None = None
        self.received_kwargs: dict[str, Any] | None = None

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        self.received_texts = list(texts)
        self.received_metadatas = metadatas
        self.received_ids = ids
        self.received_kwargs = kwargs
        return ids or []

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        return []

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncTextVectorStore:
        return cls()


def test_aadd_documents_forwards_custom_ids_once() -> None:
    vector_store = AsyncTextVectorStore()
    documents = [
        Document(page_content="Hello world", metadata={"index": 1}),
        Document(page_content="Goodbye world", metadata={"index": 2}),
    ]
    ids = ["doc_1", "doc_2"]

    result = asyncio.run(
        vector_store.aadd_documents(documents, ids=ids, namespace="test")
    )

    assert result == ids
    assert vector_store.received_texts == ["Hello world", "Goodbye world"]
    assert vector_store.received_metadatas == [{"index": 1}, {"index": 2}]
    assert vector_store.received_ids == ids
    assert vector_store.received_kwargs == {"namespace": "test"}
