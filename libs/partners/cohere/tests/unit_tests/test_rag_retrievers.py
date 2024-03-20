"""Test rag retriever integration."""


from langchain_cohere.rag_retrievers import CohereRagRetriever
from langchain_cohere.chat_models import ChatCohere


def test_initialization() -> None:
    """Test chat model initialization."""
    CohereRagRetriever(llm=ChatCohere())
