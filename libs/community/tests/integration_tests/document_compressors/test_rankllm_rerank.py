"""Test rankllm reranker."""

from langchain.retrievers.document_compressors import RankLLMRerank


def test_rankllm_reranker_init() -> None:
    """Test the RankLLM reranker initializes correctly."""
    RankLLMRerank()
