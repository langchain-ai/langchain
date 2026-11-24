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


def test_max_marginal_relevance_search_forwards_kwargs() -> None:
    """Test that max_marginal_relevance_search forwards **kwargs (#42065)."""
    texts = ["alpha", "beta", "gamma"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = Chroma.from_texts(
        collection_name="test_mmr_kwargs",
        texts=texts,
        embedding=FakeEmbeddings(size=10),
        metadatas=metadatas,
        ids=["id-0", "id-1", "id-2"],
    )
    try:
        output = docsearch.max_marginal_relevance_search(
            "alpha",
            k=2,
            fetch_k=2,
            ids=["id-1"],
        )
        assert [doc.page_content for doc in output] == ["beta"]
    finally:
        docsearch.delete_collection()
