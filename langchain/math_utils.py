"""Math utils."""
from typing import List, Union

import numpy as np

Matrix = Union[List[List[float]], List[np.ndarray], np.ndarray]


def cosine_similarity(x: Matrix, y: Matrix) -> np.ndarray:
    """Row-wise cosine similarity between two equal-width matrices."""
    x = np.array(x)
    y = np.array(y)
    if x.shape[1] != y.shape[1]:
        raise ValueError("Number of columns in x and y must be the same.")

    x_norm = np.linalg.norm(x, axis=1)
    y_norm = np.linalg.norm(y, axis=1)
    similarity = np.dot(x, y.T) / np.outer(x_norm, y_norm)
    similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
    return similarity
