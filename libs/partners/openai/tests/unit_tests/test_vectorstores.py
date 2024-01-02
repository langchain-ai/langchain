from langchain_openai.vectorstores import OpenAIVectorStore


def test_initialization() -> None:
    """Test integration vectorstore initialization."""
    OpenAIVectorStore()
