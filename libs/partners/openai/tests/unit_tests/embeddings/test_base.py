import os
from typing import List

import numpy as np
import pytest

from langchain_openai import OpenAIEmbeddings
from langchain_openai.embeddings.base import _normed_vector_avg, _vector_norm

os.environ["OPENAI_API_KEY"] = "foo"


def test_invalid_model_kwargs() -> None:
    with pytest.raises(ValueError):
        OpenAIEmbeddings(model_kwargs={"model": "foo"})


def test_incorrect_field() -> None:
    with pytest.warns(match="not default parameter"):
        llm = OpenAIEmbeddings(foo="bar")  # type: ignore[call-arg]
    assert llm.model_kwargs == {"foo": "bar"}


@pytest.mark.parametrize(
    ("vectors", "weights", "expected"),
    [
        ([[1]], [1], [1]),
        ([[1, 0, 0], [0, 1, 0]], [1, 1], [(2**0.5) * 0.5, (2**0.5) * 0.5, 0]),
        (
            [[0.27, 0.95, 0.13], [0.52, 0.1, 0.84], [0.18, 0.91, 0.36]],
            [10, 51, 3],
            [0.5235709525340383, 0.30488859567926363, 0.7955604325802826],
        ),
    ],
)
def test__normed_vector_avg(
    vectors: List[List[float]], weights: List[int], expected: List[float]
) -> None:
    actual = _normed_vector_avg(vectors, weights)
    assert np.isclose(actual, expected).all()


@pytest.mark.parametrize(
    ("vector", "expected"),
    [
        ([0.1], [1]),
        ([1, 1, 1], [3**0.5 / 3] * 3),
        ([27, 95, 13], [0.2710455418938115, 0.9536787585152627, 0.13050340905998334]),
    ],
)
def test__vector_norm(vector: List[float], expected: List[float]) -> None:
    actual = _vector_norm(vector)
    assert np.isclose(actual, expected).all()
