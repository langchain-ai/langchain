"""Test embedding model integration."""


from langchain_openai.embeddings import OpenAIEmbeddings


def test_initialization() -> None:
    """Test embedding model initialization."""
    OpenAIEmbeddings()
