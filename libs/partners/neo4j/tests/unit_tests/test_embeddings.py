"""Test embedding model integration."""

from langchain_neo4j.embeddings import Neo4jEmbeddings


def test_initialization() -> None:
    """Test embedding model initialization."""
    Neo4jEmbeddings()
