"""Integration test for embedding-based relevant doc filtering."""
import numpy as np

from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.document_compressors import EmbeddingRelevancyDocumentFilter
from langchain.retrievers.document_compressors.base import _RetrievedDocument


def test_embedding_relevant_document_filter() -> None:
    texts = [
        "What happened to all of my cookies?",
        "I wish there were better Italian restaurants in my neighborhood.",
        "My favorite color is green",
    ]
    docs = [_RetrievedDocument(page_content=t) for t in texts]
    embeddings = OpenAIEmbeddings()
    relevant_filter = EmbeddingRelevancyDocumentFilter(
        embeddings=embeddings, similarity_threshold=0.75
    )
    actual = relevant_filter.compress_documents(docs, "What did I say about food?")
    assert len(actual) == 2
    assert len(set(texts[:2]).intersection([d.page_content for d in actual])) == 2


def test_embedding_relevant_document_filter_with_metadata() -> None:
    texts = [
        "What happened to all of my cookies?",
        "I wish there were better Italian restaurants in my neighborhood.",
        "My favorite color is green",
    ]
    query = "What did I say about food?"
    embeddings = OpenAIEmbeddings()
    embedded_query = embeddings.embed_query(query)
    query_metadata = {"embedded_doc": np.zeros(len(embedded_query))}
    docs = [
        _RetrievedDocument(page_content=t, query_metadata=query_metadata) for t in texts
    ]
    docs[-1].query_metadata = {"embedded_doc": embedded_query}
    relevant_filter = EmbeddingRelevancyDocumentFilter(
        embeddings=embeddings, similarity_threshold=0.75
    )
    actual = relevant_filter.compress_documents(docs, query)
    assert len(actual) == 1
    assert texts[-1] == actual[0].page_content
