"""Test embedding model integration."""


from langchain_google_vertexai.embeddings import VertexAIEmbeddings


def test_initialization() -> None:
    """Test embedding model initialization."""
    VertexAIEmbeddings(project="fake")
