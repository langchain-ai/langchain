"""Tests for langchain_core.vectorstores.utils module."""

import math

import pytest

pytest.importorskip("numpy")
import numpy as np
import numpy.typing as npt

from langchain_core.vectorstores.utils import _cosine_similarity


class TestCosineSimilarity:
    """Tests for _cosine_similarity function."""

    def test_basic_cosine_similarity(self) -> None:
        """Test basic cosine similarity calculation."""
        # Simple orthogonal vectors
        x: list[list[float]] = [[1, 0], [0, 1]]
        y: list[list[float]] = [[1, 0], [0, 1]]
        result = _cosine_similarity(x, y)
        expected = np.array([[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_identical_vectors(self) -> None:
        """Test cosine similarity of identical vectors."""
        x: list[list[float]] = [[1, 2, 3]]
        y: list[list[float]] = [[1, 2, 3]]
        result = _cosine_similarity(x, y)
        expected = np.array([[1.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_opposite_vectors(self) -> None:
        """Test cosine similarity of opposite vectors."""
        x: list[list[float]] = [[1, 2, 3]]
        y: list[list[float]] = [[-1, -2, -3]]
        result = _cosine_similarity(x, y)
        expected = np.array([[-1.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_zero_vector(self) -> None:
        """Test cosine similarity with zero vector."""
        x: list[list[float]] = [[0, 0, 0]]
        y: list[list[float]] = [[1, 2, 3]]
        with pytest.raises(ValueError, match="NaN values found"):
            _cosine_similarity(x, y)

    def test_multiple_vectors(self) -> None:
        """Test cosine similarity with multiple vectors."""
        x: list[list[float]] = [[1, 0], [0, 1], [1, 1]]
        y: list[list[float]] = [[1, 0], [0, 1]]
        result = _cosine_similarity(x, y)
        expected = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1 / math.sqrt(2), 1 / math.sqrt(2)],
            ]
        )
        np.testing.assert_array_almost_equal(result, expected)

    def test_numpy_array_input(self) -> None:
        """Test with numpy array inputs."""
        x: npt.NDArray[np.floating] = np.array([[1, 0], [0, 1]])
        y: npt.NDArray[np.floating] = np.array([[1, 0], [0, 1]])
        result = _cosine_similarity(x, y)
        expected = np.array([[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_mixed_input_types(self) -> None:
        """Test with mixed input types (list and numpy array)."""
        x: list[list[float]] = [[1, 0], [0, 1]]
        y: npt.NDArray[np.floating] = np.array([[1, 0], [0, 1]])
        result = _cosine_similarity(x, y)
        expected = np.array([[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_higher_dimensions(self) -> None:
        """Test with higher dimensional vectors."""
        x: list[list[float]] = [[1, 0, 0, 0], [0, 1, 0, 0]]
        y: list[list[float]] = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        result = _cosine_similarity(x, y)
        expected = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_empty_matrices(self) -> None:
        """Test with empty matrices."""
        x: list[list[float]] = []
        y: list[list[float]] = []
        result = _cosine_similarity(x, y)
        expected = np.array([[]])
        np.testing.assert_array_equal(result, expected)

    def test_single_empty_matrix(self) -> None:
        """Test with one empty matrix."""
        x: list[list[float]] = []
        y: list[list[float]] = [[1, 2, 3]]
        result = _cosine_similarity(x, y)
        expected = np.array([[]])
        np.testing.assert_array_equal(result, expected)

    def test_dimension_mismatch_error(self) -> None:
        """Test error when matrices have different number of columns."""
        x: list[list[float]] = [[1, 2]]  # 2 columns
        y: list[list[float]] = [[1, 2, 3]]  # 3 columns

        with pytest.raises(
            ValueError, match="Number of columns in X and Y must be the same"
        ):
            _cosine_similarity(x, y)

    def test_nan_and_inf_handling(self) -> None:
        """Test that NaN and inf values are handled properly."""
        # Create vectors that would result in NaN/inf in similarity calculation
        x: list[list[float]] = [[0, 0]]  # Zero vector
        y: list[list[float]] = [[0, 0]]  # Zero vector
        with pytest.raises(ValueError, match="NaN values found"):
            _cosine_similarity(x, y)

    def test_large_values(self) -> None:
        """Test with large values to check numerical stability."""
        x: list[list[float]] = [[1e6, 1e6]]
        y: list[list[float]] = [[1e6, 1e6], [1e6, -1e6]]
        result = _cosine_similarity(x, y)
        expected = np.array([[1.0, 0.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_small_values(self) -> None:
        """Test with very small values."""
        x: list[list[float]] = [[1e-10, 1e-10]]
        y: list[list[float]] = [[1e-10, 1e-10], [1e-10, -1e-10]]
        result = _cosine_similarity(x, y)
        expected = np.array([[1.0, 0.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_single_vector_vs_multiple(self) -> None:
        """Test single vector against multiple vectors."""
        x: list[list[float]] = [[1, 1]]
        y: list[list[float]] = [[1, 0], [0, 1], [1, 1], [-1, -1]]
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
        x: list[list[float]] = [[5], [-3]]
        y: list[list[float]] = [[2], [-1], [4]]
        result = _cosine_similarity(x, y)
        expected = np.array(
            [
                [1.0, -1.0, 1.0],  # [5] vs [2], [-1], [4]
                [-1.0, 1.0, -1.0],  # [-3] vs [2], [-1], [4]
            ]
        )
        np.testing.assert_array_almost_equal(result, expected)
