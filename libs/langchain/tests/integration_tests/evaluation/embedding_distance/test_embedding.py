from typing import Tuple

import numpy as np
import pytest

from langchain.evaluation.embedding_distance import (
    EmbeddingDistance,
    EmbeddingDistanceEvalChain,
    PairwiseEmbeddingDistanceEvalChain,
)


@pytest.fixture
def vectors() -> Tuple[np.ndarray, np.ndarray]:
    """Create two random vectors."""
    vector_a = np.array(
        [
            0.5488135,
            0.71518937,
            0.60276338,
            0.54488318,
            0.4236548,
            0.64589411,
            0.43758721,
            0.891773,
            0.96366276,
            0.38344152,
        ]
    )
    vector_b = np.array(
        [
            0.79172504,
            0.52889492,
            0.56804456,
            0.92559664,
            0.07103606,
            0.0871293,
            0.0202184,
            0.83261985,
            0.77815675,
            0.87001215,
        ]
    )
    return vector_a, vector_b


@pytest.fixture
def pairwise_embedding_distance_eval_chain() -> PairwiseEmbeddingDistanceEvalChain:
    """Create a PairwiseEmbeddingDistanceEvalChain."""
    return PairwiseEmbeddingDistanceEvalChain()


@pytest.fixture
def embedding_distance_eval_chain() -> EmbeddingDistanceEvalChain:
    """Create a EmbeddingDistanceEvalChain."""
    return EmbeddingDistanceEvalChain()


@pytest.mark.requires("scipy")
def test_pairwise_embedding_distance_eval_chain_cosine_similarity(
    pairwise_embedding_distance_eval_chain: PairwiseEmbeddingDistanceEvalChain,
    vectors: Tuple[np.ndarray, np.ndarray],
) -> None:
    """Test the cosine similarity."""
    pairwise_embedding_distance_eval_chain.distance_metric = EmbeddingDistance.COSINE
    result = pairwise_embedding_distance_eval_chain._compute_score(np.array(vectors))
    expected = 1.0 - np.dot(vectors[0], vectors[1]) / (
        np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1])
    )
    assert np.isclose(result, expected)


@pytest.mark.requires("scipy")
def test_pairwise_embedding_distance_eval_chain_euclidean_distance(
    pairwise_embedding_distance_eval_chain: PairwiseEmbeddingDistanceEvalChain,
    vectors: Tuple[np.ndarray, np.ndarray],
) -> None:
    """Test the euclidean distance."""
    from scipy.spatial.distance import euclidean

    pairwise_embedding_distance_eval_chain.distance_metric = EmbeddingDistance.EUCLIDEAN
    result = pairwise_embedding_distance_eval_chain._compute_score(np.array(vectors))
    expected = euclidean(*vectors)
    assert np.isclose(result, expected)


@pytest.mark.requires("scipy")
def test_pairwise_embedding_distance_eval_chain_manhattan_distance(
    pairwise_embedding_distance_eval_chain: PairwiseEmbeddingDistanceEvalChain,
    vectors: Tuple[np.ndarray, np.ndarray],
) -> None:
    """Test the manhattan distance."""
    from scipy.spatial.distance import cityblock

    pairwise_embedding_distance_eval_chain.distance_metric = EmbeddingDistance.MANHATTAN
    result = pairwise_embedding_distance_eval_chain._compute_score(np.array(vectors))
    expected = cityblock(*vectors)
    assert np.isclose(result, expected)


@pytest.mark.requires("scipy")
def test_pairwise_embedding_distance_eval_chain_chebyshev_distance(
    pairwise_embedding_distance_eval_chain: PairwiseEmbeddingDistanceEvalChain,
    vectors: Tuple[np.ndarray, np.ndarray],
) -> None:
    """Test the chebyshev distance."""
    from scipy.spatial.distance import chebyshev

    pairwise_embedding_distance_eval_chain.distance_metric = EmbeddingDistance.CHEBYSHEV
    result = pairwise_embedding_distance_eval_chain._compute_score(np.array(vectors))
    expected = chebyshev(*vectors)
    assert np.isclose(result, expected)


@pytest.mark.requires("scipy")
def test_pairwise_embedding_distance_eval_chain_hamming_distance(
    pairwise_embedding_distance_eval_chain: PairwiseEmbeddingDistanceEvalChain,
    vectors: Tuple[np.ndarray, np.ndarray],
) -> None:
    """Test the hamming distance."""
    from scipy.spatial.distance import hamming

    pairwise_embedding_distance_eval_chain.distance_metric = EmbeddingDistance.HAMMING
    result = pairwise_embedding_distance_eval_chain._compute_score(np.array(vectors))
    expected = hamming(*vectors)
    assert np.isclose(result, expected)


@pytest.mark.requires("openai", "tiktoken")
def test_pairwise_embedding_distance_eval_chain_embedding_distance(
    pairwise_embedding_distance_eval_chain: PairwiseEmbeddingDistanceEvalChain,
) -> None:
    """Test the embedding distance."""
    result = pairwise_embedding_distance_eval_chain.evaluate_string_pairs(
        prediction="A single cat", prediction_b="A single cat"
    )
    assert np.isclose(result["score"], 0.0)


@pytest.mark.requires("scipy")
def test_embedding_distance_eval_chain(
    embedding_distance_eval_chain: EmbeddingDistanceEvalChain,
) -> None:
    embedding_distance_eval_chain.distance_metric = EmbeddingDistance.COSINE
    prediction = "Hi"
    reference = "Hello"
    result = embedding_distance_eval_chain.evaluate_strings(
        prediction=prediction,
        reference=reference,
    )
    assert result["score"] < 1.0
