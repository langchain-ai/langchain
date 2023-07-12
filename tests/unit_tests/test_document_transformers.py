"""Unit tests for document transformers."""
from langchain.document_transformers.embeddings_redundant_filter import (
    _filter_similar_embeddings,
)
from langchain.math_utils import cosine_similarity


def test__filter_similar_embeddings() -> None:
    threshold = 0.79
    embedded_docs = [[1.0, 2.0], [1.0, 2.0], [2.0, 1.0], [2.0, 0.5], [0.0, 0.0]]
    expected = [1, 3, 4]
    actual = _filter_similar_embeddings(embedded_docs, cosine_similarity, threshold)
    assert expected == actual


def test__filter_similar_embeddings_empty() -> None:
    assert len(_filter_similar_embeddings([], cosine_similarity, 0.0)) == 0
