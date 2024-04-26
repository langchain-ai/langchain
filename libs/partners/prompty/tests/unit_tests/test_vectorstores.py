from langchain_prompty.vectorstores import PromptyVectorStore


def test_initialization() -> None:
    """Test integration vectorstore initialization."""
    PromptyVectorStore()
