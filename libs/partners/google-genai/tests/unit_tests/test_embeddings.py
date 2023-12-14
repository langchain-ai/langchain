"""Test embeddings model integration."""


from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings


def test_integration_initialization() -> None:
    """Test chat model initialization."""
    GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key="...",
    )
    GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key="...",
        task_type="retrieval_document",
    )
