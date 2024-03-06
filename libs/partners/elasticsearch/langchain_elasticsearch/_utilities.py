from enum import Enum
from typing import List, Union

import numpy as np
from elasticsearch import Elasticsearch
from langchain_core import __version__ as langchain_version

Matrix = Union[List[List[float]], List[np.ndarray], np.ndarray]


class DistanceStrategy(str, Enum):
    """Enumerator of the Distance strategies for calculating distances
    between vectors."""

    EUCLIDEAN_DISTANCE = "EUCLIDEAN_DISTANCE"
    MAX_INNER_PRODUCT = "MAX_INNER_PRODUCT"
    DOT_PRODUCT = "DOT_PRODUCT"
    JACCARD = "JACCARD"
    COSINE = "COSINE"


def with_user_agent_header(client: Elasticsearch, header_prefix: str) -> Elasticsearch:
    headers = dict(client._headers)
    headers.update({"user-agent": f"{header_prefix}/{langchain_version}"})
    return client.options(headers=headers)


def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    embedding_list: list,
    lambda_mult: float = 0.5,
    k: int = 4,
) -> List[int]:
    """Calculate maximal marginal relevance."""
    if min(k, len(embedding_list)) <= 0:
        return []
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    similarity_to_query = cosine_similarity(query_embedding, embedding_list)[0]
    most_similar = int(np.argmax(similarity_to_query))
    idxs = [most_similar]
    selected = np.array([embedding_list[most_similar]])
    while len(idxs) < min(k, len(embedding_list)):
        best_score = -np.inf
        idx_to_add = -1
        similarity_to_selected = cosine_similarity(embedding_list, selected)
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


def cosine_similarity(X: Matrix, Y: Matrix) -> np.ndarray:
    """Row-wise cosine similarity between two equal-width matrices."""
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
        import simsimd as simd  # type: ignore

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        Z = 1 - simd.cdist(X, Y, metric="cosine")
        if isinstance(Z, float):
            return np.array([Z])
        return Z
    except ImportError:
        X_norm = np.linalg.norm(X, axis=1)
        Y_norm = np.linalg.norm(Y, axis=1)
        # Ignore divide by zero errors run time warnings as those are handled below.
        with np.errstate(divide="ignore", invalid="ignore"):
            similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)
        similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
        return similarity
