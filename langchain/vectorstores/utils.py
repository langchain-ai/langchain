"""Utility functions for working with vectors and vectorstores."""

from typing import List

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity with numpy."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    embedding_list: list,
    lambda_mult: float = 0.5,
    k: int = 4,
) -> List[int]:
    """Calculate maximal marginal relevance."""
    idxs: List[int] = []
    while len(idxs) < k:
        best_score = -np.inf
        idx_to_add = -1
        for i, emb in enumerate(embedding_list):
            if i in idxs:
                continue
            first_part = cosine_similarity(query_embedding, emb)
            second_part = 0.0
            for j in idxs:
                cos_sim = cosine_similarity(emb, embedding_list[j])
                if cos_sim > second_part:
                    second_part = cos_sim
            equation_score = lambda_mult * first_part - (1 - lambda_mult) * second_part
            if equation_score > best_score:
                best_score = equation_score
                idx_to_add = i
        idxs.append(idx_to_add)
    return idxs
