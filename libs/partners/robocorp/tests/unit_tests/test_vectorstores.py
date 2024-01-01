from langchain_robocorp.vectorstores import ActionServerVectorStore


def test_initialization() -> None:
    """Test integration vectorstore initialization."""
    ActionServerVectorStore()
