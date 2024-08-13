from langchain_databricks.vectorstores import DatabricksVectorStore


def test_initialization() -> None:
    """Test integration vectorstore initialization."""
    DatabricksVectorStore()
