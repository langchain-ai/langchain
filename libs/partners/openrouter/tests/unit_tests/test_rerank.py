"""Test OpenRouterReranker."""

from langchain_core.documents import Document
from pydantic import SecretStr

from langchain_openrouter.rerank import OpenRouterRerank


def test_initialization() -> None:
    """Test reranker initialization."""
    reranker = OpenRouterRerank(
        model_name="nvidia/llama-nemotron-rerank-vl-1b-v2:free",
        top_n=2,
        api_key=SecretStr("test-key"),
    )
    assert reranker.model == "nvidia/llama-nemotron-rerank-vl-1b-v2:free"
    assert reranker.top_n == 2


def test_serialize_documents() -> None:
    """Test document serialization."""
    docs = [
        Document(page_content="Test string"),
        Document(
            page_content="Test image", metadata={"image_url": "https://example.com"}
        ),
    ]

    serialized = OpenRouterRerank._serialize_documents(docs)

    assert len(serialized) == 2
    assert serialized[0] == "Test string"
    assert serialized[1] == {"text": "Test image", "image": "https://example.com"}
