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


class FakeImageEmbeddings:
    """Minimal embedding function that supports image embeddings."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    def embed_image(self, uris: list[str]) -> list[list[float]]:
        return [[0.4, 0.5, 0.6] for _ in uris]


def test_similarity_search_by_image_with_relevance_score() -> None:
    """Image search with scores must pass a single embedding, not a batch.

    Regression for #41528: the score variant omitted the [0] unpack that
    similarity_search_by_image has, so embed_image's list-of-lists reached
    the vector query as a nested embedding.
    """
    from langchain_core.documents import Document

    store = Chroma.from_documents(
        [Document(page_content="red car", metadata={"source": "image"})],
        collection_name="test_collection_image_scores",
        embedding=FakeImageEmbeddings(),
    )
    try:
        # Sibling without scores works: embed_image(uris=[uri])[0].
        docs = store.similarity_search_by_image("file:///tmp/query.png")
        assert len(docs) == 1

        results = store.similarity_search_by_image_with_relevance_score(
            "file:///tmp/query.png"
        )
        assert len(results) == 1
        doc, score = results[0]
        assert doc.page_content == "red car"
        assert isinstance(score, float)
    finally:
        store.delete_collection()


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
