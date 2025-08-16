"""Tests for vector store relevance score functions."""

import math

from langchain_core.vectorstores.base import VectorStore


class TestRelevanceScoreFunctions:
    """Tests for the relevance score transformation functions."""

    def test_cosine_relevance_score_fn_with_similarities(self) -> None:
        """Test cosine relevance score function with cosine similarity inputs.

        This tests the fix for GitHub issue #32498 where FAISS IndexFlatIP returns
        cosine similarity values rather than cosine distance values.
        """
        score_fn = VectorStore._cosine_relevance_score_fn

        # Test cosine similarity values (the common case for FAISS IndexFlatIP)
        test_cases = [
            (1.0, 1.0),  # Perfect similarity -> max relevance
            (0.0, 0.5),  # Orthogonal vectors -> medium relevance
            (-1.0, 0.0),  # Opposite vectors -> min relevance
            (0.5, 0.75),  # High similarity -> high relevance
            (-0.5, 0.25),  # Low similarity -> low relevance
        ]

        for similarity, expected_relevance in test_cases:
            result = score_fn(similarity)
            assert abs(result - expected_relevance) < 1e-10, (
                f"For similarity {similarity}, expected {expected_relevance}, "
                f"got {result}"
            )

    def test_cosine_relevance_score_github_issue_case(self) -> None:
        """Test the specific case mentioned in GitHub issue #32498."""
        score_fn = VectorStore._cosine_relevance_score_fn

        # The GitHub issue mentions manual cosine similarity of 0.6834729319833003
        # With the fix, this should give a high relevance score
        manual_cosine_sim = 0.6834729319833003
        relevance_score = score_fn(manual_cosine_sim)

        # Expected: (0.6834729319833003 + 1.0) / 2.0 = 0.8417364659916502
        expected = (manual_cosine_sim + 1.0) / 2.0

        assert abs(relevance_score - expected) < 1e-10

        # The relevance score should be high (> 0.5) for this high similarity
        assert relevance_score > 0.8

    def test_cosine_relevance_score_fn_properties(self) -> None:
        """Test mathematical properties of the cosine relevance score function."""
        score_fn = VectorStore._cosine_relevance_score_fn

        # Test monotonicity: higher similarity should give higher relevance
        similarities = [-1.0, -0.5, 0.0, 0.5, 1.0]
        relevance_scores = [score_fn(sim) for sim in similarities]

        # Check that scores are monotonically increasing
        for i in range(1, len(relevance_scores)):
            assert relevance_scores[i] >= relevance_scores[i - 1], (
                f"Relevance scores should be monotonically increasing: "
                f"{relevance_scores}"
            )

        # Test range: all scores should be in [0, 1]
        for sim in similarities:
            score = score_fn(sim)
            assert 0.0 <= score <= 1.0, (
                f"Relevance score {score} for similarity {sim} is outside [0, 1]"
            )

        # Test symmetry property: sim and -sim should be symmetric around 0.5
        for sim in [0.2, 0.5, 0.8]:
            pos_score = score_fn(sim)
            neg_score = score_fn(-sim)

            # The scores should be symmetric around 0.5
            assert abs((pos_score + neg_score) - 1.0) < 1e-10, (
                f"Scores for {sim} and {-sim} should be symmetric around 0.5: "
                f"{pos_score} + {neg_score} = {pos_score + neg_score}"
            )

    def test_euclidean_relevance_score_fn(self) -> None:
        """Test euclidean relevance score function for comparison."""
        score_fn = VectorStore._euclidean_relevance_score_fn

        # Test known values
        test_cases = [
            (0.0, 1.0),  # Perfect match
            (math.sqrt(2), 0.0),  # Maximum distance
            (math.sqrt(2) / 2, 0.5),  # Half distance
        ]

        for distance, expected_relevance in test_cases:
            result = score_fn(distance)
            assert abs(result - expected_relevance) < 1e-10, (
                f"For distance {distance}, expected {expected_relevance}, got {result}"
            )

    def test_max_inner_product_relevance_score_fn(self) -> None:
        """Test max inner product relevance score function for comparison."""
        score_fn = VectorStore._max_inner_product_relevance_score_fn

        # Test known values based on the actual implementation:
        # if distance > 0: return 1.0 - distance
        # else: return -1.0 * distance
        test_cases = [
            (0.0, 0.0),  # distance = 0 -> -1.0 * 0 = 0.0
            (0.5, 0.5),  # distance > 0 -> 1.0 - 0.5 = 0.5
            (-0.5, 0.5),  # distance < 0 -> -1.0 * (-0.5) = 0.5
            (-1.0, 1.0),  # distance < 0 -> -1.0 * (-1.0) = 1.0
            (1.0, 0.0),  # distance > 0 -> 1.0 - 1.0 = 0.0
        ]

        for distance, expected_relevance in test_cases:
            result = score_fn(distance)
            assert abs(result - expected_relevance) < 1e-10, (
                f"For distance {distance}, expected {expected_relevance}, got {result}"
            )

    def test_cosine_vs_other_score_functions(self) -> None:
        """Test that cosine score function behaves differently from others."""
        cosine_fn = VectorStore._cosine_relevance_score_fn
        euclidean_fn = VectorStore._euclidean_relevance_score_fn
        mip_fn = VectorStore._max_inner_product_relevance_score_fn

        # For the same input value, different functions should generally give
        # different results
        test_value = 0.5

        cosine_score = cosine_fn(test_value)
        euclidean_score = euclidean_fn(test_value)
        mip_score = mip_fn(test_value)

        # Cosine treats 0.5 as similarity: (0.5 + 1) / 2 = 0.75
        assert abs(cosine_score - 0.75) < 1e-10

        # Euclidean treats 0.5 as distance: 1 - 0.5/sqrt(2) ≈ 0.646
        expected_euclidean = 1.0 - 0.5 / math.sqrt(2)
        assert abs(euclidean_score - expected_euclidean) < 1e-10

        # MIP treats 0.5 as distance: 1 - 0.5 = 0.5
        assert abs(mip_score - 0.5) < 1e-10

        # All three should be different
        assert cosine_score != euclidean_score
        assert cosine_score != mip_score
        assert euclidean_score != mip_score
