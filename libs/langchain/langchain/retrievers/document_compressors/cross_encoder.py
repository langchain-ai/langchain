from abc import ABC, abstractmethod


class BaseCrossEncoder(ABC):
    """Interface for cross encoder models."""

    @abstractmethod
    def score(self, text_pairs: list[tuple[str, str]]) -> list[float]:
        """Score pairs' similarity.

        Args:
            text_pairs: List of pairs of texts.

        Returns:
            List of scores.
        """
