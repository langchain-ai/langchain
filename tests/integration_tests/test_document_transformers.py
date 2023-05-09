"""Integration test for embedding-based redundant doc filtering."""
from langchain.document_transformers import (
    EmbeddingsRedundantFilter,
    _DocumentWithState,
)
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document


def test_embeddings_redundant_filter() -> None:
    texts = [
        "What happened to all of my cookies?",
        "Where did all of my cookies go?",
        "I wish there were better Italian restaurants in my neighborhood.",
    ]
    docs = [Document(page_content=t) for t in texts]
    embeddings = OpenAIEmbeddings()
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    actual = redundant_filter.transform_documents(docs)
    assert len(actual) == 2
    assert set(texts[:2]).intersection([d.page_content for d in actual])


def test_embeddings_redundant_filter_with_state() -> None:
    texts = ["What happened to all of my cookies?", "foo bar baz"]
    state = {"embedded_doc": [0.5] * 10}
    docs = [_DocumentWithState(page_content=t, state=state) for t in texts]
    embeddings = OpenAIEmbeddings()
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    actual = redundant_filter.transform_documents(docs)
    assert len(actual) == 1
