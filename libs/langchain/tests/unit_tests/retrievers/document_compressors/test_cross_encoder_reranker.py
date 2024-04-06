"""Integration test for CrossEncoderReranker."""
from typing import List

from langchain_community.cross_encoders import FakeCrossEncoder
from langchain_core.documents import Document

from langchain.retrievers.document_compressors import CrossEncoderReranker


def test_rerank() -> None:
    texts = [
        "aaa1",
        "bbb1",
        "aaa2",
        "bbb2",
        "aaa3",
        "bbb3",
    ]
    docs = list(map(lambda text: Document(page_content=text), texts))
    compressor = CrossEncoderReranker(model=FakeCrossEncoder())
    actual_docs = compressor.compress_documents(docs, "bbb2")
    actual = list(map(lambda doc: doc.page_content, actual_docs))
    expected_returned = ["bbb2", "bbb1", "bbb3"]
    expected_not_returned = ["aaa1", "aaa2", "aaa3"]
    assert all([text in actual for text in expected_returned])
    assert all([text not in actual for text in expected_not_returned])
    assert actual[0] == "bbb2"


def test_rerank_empty() -> None:
    docs: List[Document] = []
    compressor = CrossEncoderReranker(model=FakeCrossEncoder())
    actual_docs = compressor.compress_documents(docs, "query")
    assert len(actual_docs) == 0
