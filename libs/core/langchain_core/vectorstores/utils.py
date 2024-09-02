"""Internal utilities for the in memory implementation of VectorStore.

These are part of a private API, and users should not use them directly
as they can change without notice.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Union

if TYPE_CHECKING:
    import numpy as np

    Matrix = Union[List[List[float]], List[np.ndarray], np.ndarray]

logger = logging.getLogger(__name__)


def _cosine_similarity(X: Matrix, Y: Matrix) -> np.ndarray:
    """Row-wise cosine similarity between two equal-width matrices.

    Args:
        X: A matrix of shape (n, m).
        Y: A matrix of shape (k, m).

    Returns:
        A matrix of shape (n, k) where each element (i, j) is the cosine similarity
        between the ith row of X and the jth row of Y.

    Raises:
        ValueError: If the number of columns in X and Y are not the same.
        ImportError: If numpy is not installed.
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "cosine_similarity requires numpy to be installed. "
            "Please install numpy with `pip install numpy`."
        ) from e

    if len(X) == 0 or len(Y) == 0:
        return np.array([])

    X = np.array(X)
    Y = np.array(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Number of columns in X and Y must be the same. X has shape {X.shape} "
            f"and Y has shape {Y.shape}."
        )
    try:
        import simsimd as simd

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        Z = 1 - np.array(simd.cdist(X, Y, metric="cosine"))
        return Z
    except ImportError:
        logger.debug(
            "Unable to import simsimd, defaulting to NumPy implementation. If you want "
            "to use simsimd please install with `pip install simsimd`."
        )
        X_norm = np.linalg.norm(X, axis=1)
        Y_norm = np.linalg.norm(Y, axis=1)
        # Ignore divide by zero errors run time warnings as those are handled below.
        with np.errstate(divide="ignore", invalid="ignore"):
            similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)
        similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
        return similarity


def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    embedding_list: list,
    lambda_mult: float = 0.5,
    k: int = 4,
) -> List[int]:
    """Calculate maximal marginal relevance.

    Args:
        query_embedding: The query embedding.
        embedding_list: A list of embeddings.
        lambda_mult: The lambda parameter for MMR. Default is 0.5.
        k: The number of embeddings to return. Default is 4.

    Returns:
        A list of indices of the embeddings to return.

    Raises:
        ImportError: If numpy is not installed.
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "maximal_marginal_relevance requires numpy to be installed. "
            "Please install numpy with `pip install numpy`."
        ) from e

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
