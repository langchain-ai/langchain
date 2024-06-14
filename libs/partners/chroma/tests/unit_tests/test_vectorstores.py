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
