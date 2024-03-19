from unittest.mock import Mock

from langchain_chroma.vectorstores import Chroma


def test_initialization() -> None:
    """Test integration vectorstore initialization."""
    # mock index
    texts = ["foo", "bar", "baz"]
    Chroma.from_texts(
        collection_name="test_collection", texts=texts, embedding=Mock()
    )
