"""Math utils."""
from typing import List, Union

import numpy as np

Matrix = Union[List[List[float]], List[np.ndarray], np.ndarray]


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

    X_norm = np.linalg.norm(X, axis=1)
    Y_norm = np.linalg.norm(Y, axis=1)
    similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)
    similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
    return similarity


def get_top_k_cosine_similarity(v1: Matrix, v2: Matrix, top_k=5, threshold_score=0.9) -> List[List[tuple]]:
    """ Row-wise cosine similarity between two equal-width matrices and return the max top_k score and index, the score all greater than threshold_score
    :param v1: matrix
    :param v2: matrix
    :param top_k: int, top k score
    :param threshold_score: float
    :return: list of index and score tuple, just like: [[(index, score),...], ...]
    
    Example:
        ## test input
        x = [[1, 2, 3, 4], [1, 2, 2, 2]]
        y = [[1, 2, 3, 5], [1, 2, 9, 5], [2, 2, 3, 5]]
        index_score_list = get_top_k_cosine_similarity(x, y, top_k=2, threshold_score=0.94)
        print('index_score_list：', index_score_list)
        ## output result
        index_score_list： [[(0, 0.9939990885479664), (2, 0.9860132971832692)], [(2, 0.9415130835240085)]]
    """
    score_array = cosine_similarity(np.array(v1), np.array(v2))
    index_score_list = []
    for i in range(score_array.shape[0]):
        # get the score which greater than threshold_score
        row = score_array[i]
        indices = np.argwhere(row > threshold_score).flatten()
        values = row[indices]
        # sort the value order by the score
        sorted_order = np.argsort(values)[::-1]
        # keep the max top k items
        if len(sorted_order) > top_k:
            sorted_order = sorted_order[:top_k]
        sorted_indices = indices[sorted_order]
        sorted_values = values[sorted_order]
        # create tuple which contains the index and value score
        tuples = list(zip(sorted_indices, sorted_values))
        index_score_list.append(tuples)
    return index_score_list
