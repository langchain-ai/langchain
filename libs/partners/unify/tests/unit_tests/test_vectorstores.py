from langchain_unify.vectorstores import UnifyVectorStore


def test_initialization() -> None:
    """Test integration vectorstore initialization."""
    UnifyVectorStore()
