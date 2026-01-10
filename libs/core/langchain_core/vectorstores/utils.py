"""Internal utilities for the in memory implementation of `VectorStore`.

These are part of a private API, and users should not use them directly
as they can change without notice.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, cast

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    import simsimd as simd  # type: ignore[import-not-found]

    _HAS_SIMSIMD = True
except ImportError:
    _HAS_SIMSIMD = False

if TYPE_CHECKING:
    Matrix = list[list[float]] | list[np.ndarray] | np.ndarray

logger = logging.getLogger(__name__)


def _cosine_similarity(x: Matrix, y: Matrix) -> np.ndarray:
    """Row-wise cosine similarity between two equal-width matrices.

    Args:
        x: A matrix of shape (n, m).
        y: A matrix of shape (k, m).

    Returns:
        A matrix of shape (n, k) where each element (i, j) is the cosine similarity
        between the ith row of X and the jth row of Y.

    Raises:
        ValueError: If the number of columns in X and Y are not the same.
        ImportError: If numpy is not installed.
    """
    if not _HAS_NUMPY:
        msg = (
            "cosine_similarity requires numpy to be installed. "
            "Please install numpy with `pip install numpy`."
        )
        raise ImportError(msg)

    if len(x) == 0 or len(y) == 0:
        return np.array([[]])

    x = np.array(x)
    y = np.array(y)

    # Check for NaN
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        warnings.warn(
            "NaN found in input arrays, unexpected return might follow",
            category=RuntimeWarning,
            stacklevel=2,
        )

    # Check for Inf
    if np.any(np.isinf(x)) or np.any(np.isinf(y)):
        warnings.warn(
            "Inf found in input arrays, unexpected return might follow",
            category=RuntimeWarning,
            stacklevel=2,
        )

    if x.shape[1] != y.shape[1]:
        msg = (
            f"Number of columns in X and Y must be the same. X has shape {x.shape} "
            f"and Y has shape {y.shape}."
        )
        raise ValueError(msg)
    if not _HAS_SIMSIMD:
        logger.debug(
            "Unable to import simsimd, defaulting to NumPy implementation. If you want "
            "to use simsimd please install with `pip install simsimd`."
        )
        x_norm = np.linalg.norm(x, axis=1)
        y_norm = np.linalg.norm(y, axis=1)
        # Ignore divide by zero errors run time warnings as those are handled below.
        with np.errstate(divide="ignore", invalid="ignore"):
            similarity = np.dot(x, y.T) / np.outer(x_norm, y_norm)
        if np.isnan(similarity).all():
            msg = "NaN values found, please remove the NaN values and try again"
            raise ValueError(msg) from None
        similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
        return cast("np.ndarray", similarity)

    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return 1 - np.array(simd.cdist(x, y, metric="cosine"))


def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    embedding_list: list,
    lambda_mult: float = 0.5,
    k: int = 4,
) -> list[int]:
    """Calculate maximal marginal relevance.

    Args:
        query_embedding: The query embedding.
        embedding_list: A list of embeddings.
        lambda_mult: The lambda parameter for MMR.
        k: The number of embeddings to return.

    Returns:
        A list of indices of the embeddings to return.

    Raises:
        ImportError: If numpy is not installed.
    """
    if not _HAS_NUMPY:
        msg = (
            "maximal_marginal_relevance requires numpy to be installed. "
            "Please install numpy with `pip install numpy`."
        )
        raise ImportError(msg)

    if min(k, len(embedding_list)) <= 0:
        return []
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    similarity_to_query = _cosine_similarity(query_embedding, embedding_list)[0]
    most_similar = int(np.argmax(similarity_to_query))
    idxs = [most_similar]
    selected = np.array([embedding_list[most_similar]])
    while len(idxs) < min(k, len(embedding_list)):
        best_score = -np.inf
        idx_to_add = -1
        similarity_to_selected = _cosine_similarity(embedding_list, selected)
        for i, query_score in enumerate(similarity_to_query):
            if i in idxs:
                continue
            redundant_score = max(similarity_to_selected[i])
            equation_score = (
                lambda_mult * query_score - (1 - lambda_mult) * redundant_score
            )
            if equation_score > best_score:
                best_score = equation_score
                idx_to_add = i
        idxs.append(idx_to_add)
        selected = np.append(selected, [embedding_list[idx_to_add]], axis=0)
    return idxs
