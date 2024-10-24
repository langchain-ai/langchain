from langchain_neo4j.vectorstores import Neo4jVectorStore


def test_initialization() -> None:
    """Test integration vectorstore initialization."""
    Neo4jVectorStore()
