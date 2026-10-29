import pytest
from langchain_core.embeddings.fake import (
    DeterministicFakeEmbedding,
    FakeEmbeddings,
)
from pytest_mock import MockerFixture

from langchain_chroma.vectorstores import Chroma


class FakeImageEmbeddings(DeterministicFakeEmbedding):
    """Fake embeddings that also embed images, one vector per URI."""

    def embed_image(self, uris: list[str]) -> list[list[float]]:
        return [self.embed_query(uri) for uri in uris]


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


def test_similarity_search_by_image_with_relevance_score(
    mocker: MockerFixture,
) -> None:
    """Test that a single flat image embedding is used as the query vector."""
    embedding = FakeImageEmbeddings(size=10)
    docsearch = Chroma.from_texts(
        collection_name="test_image_relevance_collection",
        texts=["foo", "bar", "baz"],
        embedding=embedding,
    )
    spy = mocker.spy(docsearch, "similarity_search_by_vector_with_relevance_scores")
    output = docsearch.similarity_search_by_image_with_relevance_score(uri="foo", k=1)
    docsearch.delete_collection()
    assert spy.call_args.kwargs["embedding"] == embedding.embed_image(uris=["foo"])[0]
    assert len(output) == 1
    doc, score = output[0]
    assert doc.page_content == "foo"
    assert score == pytest.approx(0.0, abs=1e-6)
