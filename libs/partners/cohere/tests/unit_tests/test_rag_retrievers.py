"""Test rag retriever integration."""


from langchain_cohere.chat_models import ChatCohere
from langchain_cohere.rag_retrievers import CohereRagRetriever


def test_initialization() -> None:
    """Test chat model initialization."""
    CohereRagRetriever(llm=ChatCohere(cohere_api_key="test"))
