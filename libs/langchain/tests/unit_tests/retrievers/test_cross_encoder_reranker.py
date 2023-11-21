"""Integration test for CrossEncoderReranker."""
from langchain.cross_encoders import FakeCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.schema import Document


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
    docs = []
    compressor = CrossEncoderReranker(model=FakeCrossEncoder())
    actual_docs = compressor.compress_documents(docs, "query")
    assert len(actual_docs) == 0
