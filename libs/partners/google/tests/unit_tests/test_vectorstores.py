from google.vectorstores import GoogleGenerativeAIChatVectorStore


def test_integration_vectorstore_initialization() -> None:
    """Test integration vectorstore initialization."""
    GoogleGenerativeAIChatVectorStore()
