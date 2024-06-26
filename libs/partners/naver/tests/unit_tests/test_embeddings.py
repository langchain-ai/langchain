"""Test embedding model integration."""


from langchain_naver.embeddings import NaverEmbeddings


def test_initialization() -> None:
    """Test embedding model initialization."""
    NaverEmbeddings()
