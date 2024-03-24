"""Test the cohere reranker."""
import os

from langchain.retrievers.document_compressors.cohere_rerank import CohereRerank

os.environ["COHERE_API_KEY"] = "foo"


def test_cohere_reranker_init() -> None:
    """Test the cohere reranker initializes correctly."""
    CohereRerank()
