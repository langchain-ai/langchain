"""Test embedding model integration."""

from langchain_databricks.embeddings import DatabricksEmbeddings


def test_initialization() -> None:
    """Test embedding model initialization."""
    DatabricksEmbeddings()
