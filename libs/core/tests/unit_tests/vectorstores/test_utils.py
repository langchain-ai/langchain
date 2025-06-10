"""Tests for langchain_core.vectorstores.utils module."""

import math

import pytest

pytest.importorskip("numpy")
import numpy as np

from langchain_core.vectorstores.utils import _cosine_similarity


class TestCosineSimilarity:
    """Tests for _cosine_similarity function."""

    def test_basic_cosine_similarity(self) -> None:
        """Test basic cosine similarity calculation."""
        # Simple orthogonal vectors
        x = [[1, 0], [0, 1]]
        y = [[1, 0], [0, 1]]
        result = _cosine_similarity(x, y)
        expected = np.array([[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_identical_vectors(self) -> None:
        """Test cosine similarity of identical vectors."""
        x = [[1, 2, 3]]
        y = [[1, 2, 3]]
        result = _cosine_similarity(x, y)
        expected = np.array([[1.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_opposite_vectors(self) -> None:
        """Test cosine similarity of opposite vectors."""
        x = [[1, 2, 3]]
        y = [[-1, -2, -3]]
        result = _cosine_similarity(x, y)
        expected = np.array([[-1.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_zero_vector(self) -> None:
        """Test cosine similarity with zero vector."""
        x = [[0, 0, 0]]
        y = [[1, 2, 3]]
        result = _cosine_similarity(x, y)
        expected = np.array([[0.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_multiple_vectors(self) -> None:
        """Test cosine similarity with multiple vectors."""
        x = [[1, 0], [0, 1], [1, 1]]
        y = [[1, 0], [0, 1]]
        result = _cosine_similarity(x, y)

        # Expected:
        # [1,0] vs [1,0] = 1.0, [1,0] vs [0,1] = 0.0
        # [0,1] vs [1,0] = 0.0, [0,1] vs [0,1] = 1.0
        # [1,1] vs [1,0] = 1/√2, [1,1] vs [0,1] = 1/√2
        expected = np.array(
            [[1.0, 0.0], [0.0, 1.0], [1 / math.sqrt(2), 1 / math.sqrt(2)]]
        )
        np.testing.assert_array_almost_equal(result, expected)

    def test_numpy_array_input(self) -> None:
        """Test with numpy array inputs."""
        x = np.array([[1, 0], [0, 1]])
        y = np.array([[1, 0], [0, 1]])
        result = _cosine_similarity(x, y)
        expected = np.array([[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_mixed_input_types(self) -> None:
        """Test with mixed input types (list and numpy array)."""
        x = [[1, 0], [0, 1]]
        y = np.array([[1, 0], [0, 1]])
        result = _cosine_similarity(x, y)
        expected = np.array([[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_higher_dimensions(self) -> None:
        """Test with higher dimensional vectors."""
        x = [[1, 0, 0, 0], [0, 1, 0, 0]]
        y = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        result = _cosine_similarity(x, y)
        expected = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_empty_matrices(self) -> None:
        """Test with empty matrices."""
        x = []
        y = []
        result = _cosine_similarity(x, y)
        expected = np.array([[]])
        np.testing.assert_array_equal(result, expected)

    def test_single_empty_matrix(self) -> None:
        """Test with one empty matrix."""
        x = []
        y = [[1, 2, 3]]
        result = _cosine_similarity(x, y)
        expected = np.array([[]])
        np.testing.assert_array_equal(result, expected)

    def test_dimension_mismatch_error(self) -> None:
        """Test error when matrices have different number of columns."""
        x = [[1, 2]]  # 2 columns
        y = [[1, 2, 3]]  # 3 columns

        with pytest.raises(
            ValueError, match="Number of columns in X and Y must be the same"
        ):
            _cosine_similarity(x, y)

    def test_nan_and_inf_handling(self) -> None:
        """Test that NaN and inf values are handled properly."""
        # Create vectors that would result in NaN/inf in similarity calculation
        x = [[0, 0]]  # Zero vector
        y = [[0, 0]]  # Zero vector
        result = _cosine_similarity(x, y)

        # Should return 0.0 instead of NaN
        expected = np.array([[0.0]])
        np.testing.assert_array_equal(result, expected)
        assert not np.isnan(result).any()
        assert not np.isinf(result).any()

    def test_large_values(self) -> None:
        """Test with large values to check numerical stability."""
        x = [[1e6, 1e6]]
        y = [[1e6, 1e6], [1e6, -1e6]]
        result = _cosine_similarity(x, y)
        expected = np.array([[1.0, 0.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_small_values(self) -> None:
        """Test with very small values."""
        x = [[1e-10, 1e-10]]
        y = [[1e-10, 1e-10], [1e-10, -1e-10]]
        result = _cosine_similarity(x, y)
        expected = np.array([[1.0, 0.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_single_vector_vs_multiple(self) -> None:
        """Test single vector against multiple vectors."""
        x = [[1, 1]]
        y = [[1, 0], [0, 1], [1, 1], [-1, -1]]
        result = _cosine_similarity(x, y)
        expected = np.array(
            [
                [
                    1 / math.sqrt(2),  # cos(45°)
                    1 / math.sqrt(2),  # cos(45°)
                    1.0,  # cos(0°)
                    -1.0,  # cos(180°)
                ]
            ]
        )
        np.testing.assert_array_almost_equal(result, expected)

    def test_single_dimension_vectors(self) -> None:
        """Test with single-dimension vectors."""
        x = [[5], [-3]]
        y = [[2], [-1], [4]]
        result = _cosine_similarity(x, y)
        expected = np.array(
            [
                [1.0, -1.0, 1.0],  # [5] vs [2], [-1], [4]
                [-1.0, 1.0, -1.0],  # [-3] vs [2], [-1], [4]
            ]
        )
        np.testing.assert_array_almost_equal(result, expected)
