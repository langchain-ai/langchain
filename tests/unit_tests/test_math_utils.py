"""Test math utility functions."""
from typing import List

import numpy as np
import pytest

from langchain.math_utils import cosine_similarity, cosine_similarity_top_k


@pytest.fixture
def X() -> List[List[float]]:
    return [[1.0, 2.0, 3.0], [0.0, 1.0, 0.0], [1.0, 2.0, 0.0]]


@pytest.fixture
def Y() -> List[List[float]]:
    return [[0.5, 1.0, 1.5], [1.0, 0.0, 0.0], [2.0, 5.0, 2.0], [0.0, 0.0, 0.0]]


def test_cosine_similarity_zero() -> None:
    X = np.zeros((3, 3))
    Y = np.random.random((3, 3))
    expected = np.zeros((3, 3))
    actual = cosine_similarity(X, Y)
    assert np.allclose(expected, actual)


def test_cosine_similarity_identity() -> None:
    X = np.random.random((4, 4))
    expected = np.ones(4)
    actual = np.diag(cosine_similarity(X, X))
    assert np.allclose(expected, actual)


def test_cosine_similarity_empty() -> None:
    empty_list: List[List[float]] = []
    assert len(cosine_similarity(empty_list, empty_list)) == 0
    assert len(cosine_similarity(empty_list, np.random.random((3, 3)))) == 0


def test_cosine_similarity(X: List[List[float]], Y: List[List[float]]) -> None:
    expected = [
        [1.0, 0.26726124, 0.83743579, 0.0],
        [0.53452248, 0.0, 0.87038828, 0.0],
        [0.5976143, 0.4472136, 0.93419873, 0.0],
    ]
    actual = cosine_similarity(X, Y)
    assert np.allclose(expected, actual)


def test_cosine_similarity_top_k(X: List[List[float]], Y: List[List[float]]) -> None:
    expected_idxs = [(0, 0), (2, 2), (1, 2), (0, 2), (2, 0)]
    expected_scores = [1.0, 0.93419873, 0.87038828, 0.83743579, 0.5976143]
    actual_idxs, actual_scores = cosine_similarity_top_k(X, Y)
    assert actual_idxs == expected_idxs
    assert np.allclose(expected_scores, actual_scores)


def test_cosine_similarity_score_threshold(
    X: List[List[float]], Y: List[List[float]]
) -> None:
    expected_idxs = [(0, 0), (2, 2)]
    expected_scores = [1.0, 0.93419873]
    actual_idxs, actual_scores = cosine_similarity_top_k(
        X, Y, top_k=None, score_threshold=0.9
    )
    assert actual_idxs == expected_idxs
    assert np.allclose(expected_scores, actual_scores)


def test_cosine_similarity_top_k_and_score_threshold(
    X: List[List[float]], Y: List[List[float]]
) -> None:
    expected_idxs = [(0, 0), (2, 2), (1, 2), (0, 2)]
    expected_scores = [1.0, 0.93419873, 0.87038828, 0.83743579]
    actual_idxs, actual_scores = cosine_similarity_top_k(X, Y, score_threshold=0.8)
    assert actual_idxs == expected_idxs
    assert np.allclose(expected_scores, actual_scores)
