import math
from typing import List

import numpy as np


def sliced_norm_l2(vector: List[float], dim=2048) -> List[float]:
    """
    Adjust the dimensionality of the vector.

    Reducing the dimensionality too much can decrease the
    distinguishability of different content,
    leading to lower precision, recall, and NDCG in retrieval tasks.

    :param vector: The original vector to be resized.
    :param dim: The target dimensionality.
    :return: A vector resized to the specified dimensionality.
    """
    norm = float(np.linalg.norm(vector[:dim]))
    return [v / norm for v in vector[:dim]]


def dot_product(vec1: List[float], vec2: List[float]):
    if len(vec1) != len(vec2):
        raise ValueError("向量长度必须相同")
    return sum(a * b for a, b in zip(vec1, vec2))


def magnitude(vec: List[float]):
    return math.sqrt(sum(a**2 for a in vec))


def cosine_similarity(vec1: List[float], vec2: List[float]):
    """余弦相似度"""
    dot = dot_product(vec1, vec2)
    mag1 = magnitude(vec1)
    mag2 = magnitude(vec2)
    if mag1 == 0 or mag2 == 0:
        return 0  # 避免除零错误
    return dot / (mag1 * mag2)
