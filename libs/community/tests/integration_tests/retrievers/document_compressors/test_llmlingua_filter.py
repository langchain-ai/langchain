"""Test the LLMLingua Filter."""

from langchain_community.retrievers.document_compressors import LLMLinguaCompressor


def test_llmlingua_filter_init() -> None:
    """Test the llmlingua initializes correctly."""
    LLMLinguaCompressor()
