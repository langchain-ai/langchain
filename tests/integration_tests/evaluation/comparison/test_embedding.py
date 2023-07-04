from typing import Tuple

import numpy as np
import pytest

from langchain.evaluation.comparison.embedding import (
    EmbeddingDistance,
    PairwiseEmbeddingStringEvalChain,
)


@pytest.fixture
def vectors() -> Tuple[np.ndarray, np.ndarray]:
    """Create two random vectors."""
    np.random.seed(0)
    vector_a = np.random.rand(10)
    vector_b = np.random.rand(10)
    return vector_a, vector_b


@pytest.fixture
def chain() -> PairwiseEmbeddingStringEvalChain:
    """Create a PairwiseEmbeddingStringEvalChain."""
    return PairwiseEmbeddingStringEvalChain()


@pytest.mark.requires("scipy")
def test_cosine_similarity(
    chain: PairwiseEmbeddingStringEvalChain, vectors: Tuple[np.ndarray, np.ndarray]
) -> None:
    """Test the cosine similarity."""
    chain.distance_metric = EmbeddingDistance.COSINE
    result = chain._compute_score(np.array(vectors))
    expected = 1.0 - np.dot(vectors[0], vectors[1]) / (
        np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1])
    )
    assert np.isclose(result, expected)


@pytest.mark.requires("scipy")
def test_euclidean_distance(
    chain: PairwiseEmbeddingStringEvalChain, vectors: Tuple[np.ndarray, np.ndarray]
) -> None:
    """Test the euclidean distance."""
    from scipy.spatial.distance import euclidean

    chain.distance_metric = EmbeddingDistance.EUCLIDEAN
    result = chain._compute_score(np.array(vectors))
    expected = euclidean(*vectors)
    assert np.isclose(result, expected)


@pytest.mark.requires("scipy")
def test_manhattan_distance(
    chain: PairwiseEmbeddingStringEvalChain, vectors: Tuple[np.ndarray, np.ndarray]
) -> None:
    """Test the manhattan distance."""
    from scipy.spatial.distance import cityblock

    chain.distance_metric = EmbeddingDistance.MANHATTAN
    result = chain._compute_score(np.array(vectors))
    expected = cityblock(*vectors)
    assert np.isclose(result, expected)


@pytest.mark.requires("scipy")
def test_chebyshev_distance(
    chain: PairwiseEmbeddingStringEvalChain, vectors: Tuple[np.ndarray, np.ndarray]
) -> None:
    """Test the chebyshev distance."""
    from scipy.spatial.distance import chebyshev

    chain.distance_metric = EmbeddingDistance.CHEBYSHEV
    result = chain._compute_score(np.array(vectors))
    expected = chebyshev(*vectors)
    assert np.isclose(result, expected)


@pytest.mark.requires("scipy")
def test_hamming_distance(
    chain: PairwiseEmbeddingStringEvalChain, vectors: Tuple[np.ndarray, np.ndarray]
) -> None:
    """Test the hamming distance."""
    from scipy.spatial.distance import hamming

    chain.distance_metric = EmbeddingDistance.HAMMING
    result = chain._compute_score(np.array(vectors))
    expected = hamming(*vectors)
    assert np.isclose(result, expected)


@pytest.mark.requires("openai", "tiktoken")
def test_embedding_distance(chain: PairwiseEmbeddingStringEvalChain) -> None:
    """Test the embedding distance."""
    result = chain.evaluate_string_pairs(
        prediction="A single cat", prediction_b="A single cat"
    )
    assert np.isclose(result["score"], 0.0)
