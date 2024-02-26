from langchain_kinetica.vectorstores import KineticaVectorStore


def test_initialization() -> None:
    """Test integration vectorstore initialization."""
    KineticaVectorStore()
