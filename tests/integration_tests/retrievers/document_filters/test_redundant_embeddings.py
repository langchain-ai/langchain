"""Integration test for embedding-based redundant doc filtering."""
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.document_filters import EmbeddingRedundantDocumentFilter
from langchain.retrievers.document_filters.base import _RetrievedDocument


def test_embedding_redundant_document_filter() -> None:
    texts = [
        "What happened to all of my cookies?",
        "Where did all of my cookies go?",
        "I wish there were better Italian restaurants in my neighborhood.",
    ]
    docs = [_RetrievedDocument(page_content=t) for t in texts]
    embeddings = OpenAIEmbeddings()
    redundant_filter = EmbeddingRedundantDocumentFilter(embeddings=embeddings)
    actual = redundant_filter.filter(docs, "foo")
    assert len(actual) == 2
    assert set(texts[:2]).intersection([d.page_content for d in actual])


def test_embedding_redundant_document_filter_with_query_metadata() -> None:
    texts = ["What happened to all of my cookies?", "foo bar baz"]
    query_metadata = {"embedded_doc": [0.5] * 10}
    docs = [
        _RetrievedDocument(page_content=t, query_metadata=query_metadata) for t in texts
    ]
    embeddings = OpenAIEmbeddings()
    redundant_filter = EmbeddingRedundantDocumentFilter(embeddings=embeddings)
    actual = redundant_filter.filter(docs, "foo")
    assert len(actual) == 1
