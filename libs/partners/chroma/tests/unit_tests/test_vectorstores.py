from langchain_chroma.vectorstores import Chroma
from tests.integration_tests.fake_embeddings import (
    FakeEmbeddings,
)


def test_initialization() -> None:
    """Test integration vectorstore initialization."""
    texts = ["foo", "bar", "baz"]
    Chroma.from_texts(
        collection_name="test_collection", texts=texts, embedding=FakeEmbeddings()
    )
