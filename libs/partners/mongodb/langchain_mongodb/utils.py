"""
Tools for the Maximal Marginal Relevance (MMR) reranking.
Duplicated from langchain_community to avoid cross-dependencies.

Functions "maximal_marginal_relevance" and "cosine_similarity"
are duplicated in this utility respectively from modules:
    - "libs/community/langchain_community/vectorstores/utils.py"
    - "libs/community/langchain_community/utils/math.py"
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Dict, List, Union

import numpy as np

logger = logging.getLogger(__name__)

Matrix = Union[List[List[float]], List[np.ndarray], np.ndarray]


class FailCode:
    INDEX_NOT_FOUND = 27
    INDEX_ALREADY_EXISTS = 68


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


def str_to_oid(str_repr: str) -> Any | str:
    """Attempt to cast string representation of id to MongoDB's internal BSON ObjectId.

    To be consistent with ObjectId, input must be a 24 character hex string.
    If it is not, MongoDB will happily use the string in the main _id index.
    Importantly, the str representation that comes out of MongoDB will have this form.

    Args:
        str_repr: id as string.

    Returns:
        ObjectID
    """
    from bson import ObjectId
    from bson.errors import InvalidId

    try:
        return ObjectId(str_repr)
    except InvalidId:
        logger.debug(
            "ObjectIds must be 12-character byte or 24-character hex strings. "
            "Examples: b'heres12bytes', '6f6e6568656c6c6f68656768'"
        )
        return str_repr


def oid_to_str(oid: Any) -> str:
    """Convert MongoDB's internal BSON ObjectId into a simple str for compatibility.

    Instructive helper to show where data is coming out of MongoDB.

    Args:
        oid: bson.ObjectId

    Returns:
        24 character hex string.
    """
    return str(oid)


def make_serializable(
    obj: Dict[str, Any],
) -> None:
    """Recursively cast values in a dict to a form able to json.dump"""
    from bson import ObjectId

    for k, v in obj.items():
        if isinstance(v, dict):
            make_serializable(v)
        elif isinstance(v, list) and v and isinstance(v[0], (ObjectId, date, datetime)):
            obj[k] = [oid_to_str(item) for item in v]
        elif isinstance(v, ObjectId):
            obj[k] = oid_to_str(v)
        elif isinstance(v, (datetime, date)):
            obj[k] = v.isoformat()
