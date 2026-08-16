from langchain_core.documents import Document
from langchain_core.embeddings.fake import (
    FakeEmbeddings,
)

from langchain_chroma.vectorstores import Chroma


class FakeEmbeddingsWithImage(FakeEmbeddings):
    """FakeEmbeddings that also implements embed_image, for image search tests."""

    def embed_image(self, uris: list[str]) -> list[list[float]]:
        return [self._get_embedding() for _ in uris]


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


def test_similarity_search_by_image() -> None:
    """Test similarity search by image returns Documents."""
    texts = ["foo", "bar", "baz"]
    docsearch = Chroma.from_texts(
        collection_name="test_collection_image",
        texts=texts,
        embedding=FakeEmbeddingsWithImage(size=10),
    )
    output = docsearch.similarity_search_by_image(uri="fake_uri.jpg", k=1)
    docsearch.delete_collection()
    assert len(output) == 1
    assert all(isinstance(doc, Document) for doc in output)


def test_similarity_search_by_image_with_relevance_score() -> None:
    """Test similarity search by image with relevance score.

    Returns scored Documents without raising on the embedding shape.
    """
    texts = ["foo", "bar", "baz"]
    docsearch = Chroma.from_texts(
        collection_name="test_collection_image_score",
        texts=texts,
        embedding=FakeEmbeddingsWithImage(size=10),
    )
    output = docsearch.similarity_search_by_image_with_relevance_score(
        uri="fake_uri.jpg", k=1
    )
    docsearch.delete_collection()
    assert len(output) == 1
    doc, score = output[0]
    assert isinstance(doc, Document)
    assert isinstance(score, float)
