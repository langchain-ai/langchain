import pytest
from langchain_core.embeddings.fake import (
    FakeEmbeddings,
)

from langchain_chroma.vectorstores import Chroma


def test_initialization() -> None:
    """Test integration vectorstore initialization."""
    texts = ["foo", "bar", "baz"]
    Chroma.from_texts(
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(size=10),
    )


def test_similarity_search() -> None:
    """Test similarity search by Chroma."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = Chroma.from_texts(
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(size=10),
        metadatas=metadatas,
    )
    output = docsearch.similarity_search("foo", k=1)
    docsearch.delete_collection()
    assert len(output) == 1


def test_init_rejects_coroutine_client() -> None:
    """Passing a coroutine as `client` should fail with a clear message."""

    async def _build_client() -> object:
        return object()

    client_coro = _build_client()
    try:
        with pytest.raises(ValueError, match="`client` is a coroutine"):
            Chroma(client=client_coro)
    finally:
        client_coro.close()


def test_init_rejects_async_client_api_like_object() -> None:
    """Async Chroma-style clients should fail fast instead of raising obscure errors."""

    class _AsyncClientLike:
        async def get_or_create_collection(self, **_kwargs: object) -> object:
            return object()

    with pytest.raises(
        ValueError,
        match="Asynchronous Chroma clients are not supported",
    ):
        Chroma(client=_AsyncClientLike())
