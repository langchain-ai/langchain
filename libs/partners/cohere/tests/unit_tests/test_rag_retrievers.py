"""Test rag retriever integration."""


from langchain_cohere.rag_retrievers import CohereRagRetriever


def test_initialization() -> None:
    """Test chat model initialization."""
    CohereRagRetriever()
