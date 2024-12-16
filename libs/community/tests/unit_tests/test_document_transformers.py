"""Unit tests for document transformers."""

import pytest

pytest.importorskip("langchain_community")

from langchain_community.document_transformers.embeddings_redundant_filter import (  # noqa: E402
    _filter_similar_embeddings,
)
from langchain_community.utils.math import cosine_similarity  # noqa: E402


def test__filter_similar_embeddings() -> None:
    threshold = 0.79
    embedded_docs = [[1.0, 2.0], [1.0, 2.0], [2.0, 1.0], [2.0, 0.5], [0.0, 0.0]]
    expected = [1, 3, 4]
    actual = _filter_similar_embeddings(embedded_docs, cosine_similarity, threshold)
    assert expected == actual


def test__filter_similar_embeddings_empty() -> None:
    assert len(_filter_similar_embeddings([], cosine_similarity, 0.0)) == 0
