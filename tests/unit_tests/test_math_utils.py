"""Test math utility functions."""
from typing import List

import numpy as np

from langchain.math_utils import cosine_similarity


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


def test_cosine_similarity() -> None:
    X = [[1.0, 2.0, 3.0], [0.0, 1.0, 0.0], [1.0, 2.0, 0.0]]
    Y = [[0.5, 1.0, 1.5], [1.0, 0.0, 0.0], [2.0, 5.0, 2.0]]
    expected = [
        [1.0, 0.26726124, 0.83743579],
        [0.53452248, 0.0, 0.87038828],
        [0.5976143, 0.4472136, 0.93419873],
    ]
    actual = cosine_similarity(X, Y)
    assert np.allclose(expected, actual)
