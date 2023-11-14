"""Integration test for Kay.ai API Wrapper."""
import pytest

from langchain.retrievers import KayAiRetriever
from langchain.schema import Document


@pytest.mark.requires("kay")
def test_kay_retriever() -> None:
    retriever = KayAiRetriever.create(
        dataset_id="company",
        data_types=["10-K", "10-Q", "8-K", "PressRelease"],
        num_contexts=3,
    )
    docs = retriever.get_relevant_documents(
        "What were the biggest strategy changes and partnerships made by Roku "
        "in 2023?",
    )
    assert len(docs) == 3
    for doc in docs:
        assert isinstance(doc, Document)
        assert doc.page_content
        assert doc.metadata
        assert len(list(doc.metadata.items())) > 0
