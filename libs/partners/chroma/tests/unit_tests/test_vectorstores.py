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


class FakeImageEmbeddings(FakeEmbeddings):
    """Fake embeddings supporting image embeddings."""

    def embed_image(self, uris: list[str]) -> list[list[float]]:
        return [[0.1] * self.size for _ in uris]


def test_similarity_search_by_image() -> None:
    """Test image similarity search and relevance score search."""
    texts = ["foo", "bar", "baz"]
    embeddings = FakeImageEmbeddings(size=10)
    docsearch = Chroma.from_texts(
        collection_name="test_image_collection",
        texts=texts,
        embedding=embeddings,
    )
    docs = docsearch.similarity_search_by_image("file:///dummy.png", k=1)
    assert len(docs) == 1
    assert docs[0].page_content in texts

    scored_docs = docsearch.similarity_search_by_image_with_relevance_score(
        "file:///dummy.png", k=1
    )
    docsearch.delete_collection()
    assert len(scored_docs) == 1
    doc, score = scored_docs[0]
    assert doc.page_content in texts
    assert isinstance(score, float)
