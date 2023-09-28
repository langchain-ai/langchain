"""Integration test for embedding-based relevant doc filtering."""
import numpy as np

from langchain.document_transformers.embeddings_redundant_filter import (
    _DocumentWithState,
)
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.schema import Document


def test_embeddings_filter() -> None:
    texts = [
        "What happened to all of my cookies?",
        "I wish there were better Italian restaurants in my neighborhood.",
        "My favorite color is green",
    ]
    docs = [Document(page_content=t) for t in texts]
    embeddings = OpenAIEmbeddings()
    relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.75)
    actual = relevant_filter.compress_documents(docs, "What did I say about food?")
    assert len(actual) == 2
    assert len(set(texts[:2]).intersection([d.page_content for d in actual])) == 2


def test_embeddings_filter_with_state() -> None:
    texts = [
        "What happened to all of my cookies?",
        "I wish there were better Italian restaurants in my neighborhood.",
        "My favorite color is green",
    ]
    query = "What did I say about food?"
    embeddings = OpenAIEmbeddings()
    embedded_query = embeddings.embed_query(query)
    state = {"embedded_doc": np.zeros(len(embedded_query))}
    docs = [_DocumentWithState(page_content=t, state=state) for t in texts]
    docs[-1].state = {"embedded_doc": embedded_query}
    relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.75)
    actual = relevant_filter.compress_documents(docs, query)
    assert len(actual) == 1
    assert texts[-1] == actual[0].page_content
