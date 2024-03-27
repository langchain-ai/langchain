from abc import ABC, abstractmethod
from typing import List


class CrossEncoder(ABC):
    """Interface for cross encoder models."""

    @abstractmethod
    def score(self, pairs: List[List[str]]) -> List[float]:
        """Score pairs' similarity.

        Args:
            pairs: List of pairs of texts.

        Returns:
            List of scores.
        """
