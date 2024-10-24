"""Test Neo4j Chat API wrapper."""

from langchain_neo4j import Neo4jLLM


def test_initialization() -> None:
    """Test integration initialization."""
    Neo4jLLM()
