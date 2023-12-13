from nvidia_aiplay.vectorstores import NVAIPlayVectorStore


def test_integration_vectorstore_initialization() -> None:
    """Test integration vectorstore initialization."""
    NVAIPlayVectorStore()
