"""Test Cohere reranks."""
from langchain_core.documents import Document

from langchain_cohere import CohereRerank


def test_langchain_cohere_rerank_documents() -> None:
    """Test cohere rerank."""
    rerank = CohereRerank()
    test_documents = [
        Document(page_content="This is a test document."),
        Document(page_content="Another test document."),
    ]
    test_query = "Test query"
    results = rerank.rerank(test_documents, test_query)
    assert len(results) == 2
