"""Integration tests for OpenRouterReranker."""

from langchain_core.documents import Document

from langchain_openrouter.rerank import OpenRouterRerank


def test_compress_documents() -> None:
    """Test compressing documents."""
    reranker = OpenRouterRerank(
        model_name="nvidia/llama-nemotron-rerank-vl-1b-v2:free",
        top_n=2,
    )

    docs = [
        Document(
            page_content="LangChain is a framework for developing applications "
            "powered by language models via openrouter."
        ),
        Document(page_content="The weather in Tokyo is 72 degrees and sunny."),
        Document(
            page_content="OpenRouter provides a unified API for interacting "
            "with dozens of LLMs."
        ),
        Document(
            page_content="Cats are popular pets and are known for their independence."
        ),
    ]

    query = "What is OpenRouter and how does it relate to LLMs?"

    ranked_docs = reranker.compress_documents(documents=docs, query=query)

    assert len(ranked_docs) <= 2
    assert ranked_docs[0].page_content == (
        "OpenRouter provides a unified API for interacting with dozens of LLMs."
    )
    assert "relevance_score" in ranked_docs[0].metadata
