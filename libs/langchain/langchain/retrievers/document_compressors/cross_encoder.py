from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseCrossEncoder(ABC):
    """Interface for cross encoder models."""

    @abstractmethod
    def score(self, text_pairs: List[Tuple[str, str]]) -> List[float]:
        """Score pairs' similarity.

        Args:
            text_pairs: List of pairs of texts.

        Returns:
            List of scores.
        """
