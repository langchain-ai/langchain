"""Cross Encoder interface.

Cross encoders are models that score the similarity between pairs of texts.
Unlike bi-encoders that encode texts independently, cross encoders process
both texts together, often achieving better accuracy for tasks like
semantic similarity and reranking.

Example:
    .. code-block:: python

        class MyCrossEncoder(BaseCrossEncoder):
            def score(self, text_pairs):
                # Return similarity scores for each pair
                return [0.9, 0.3, 0.8]

        encoder = MyCrossEncoder()
        scores = encoder.score([
            ("query1", "document1"),
            ("query1", "document2"),
        ])
"""

from abc import ABC, abstractmethod


class BaseCrossEncoder(ABC):
    """Abstract interface for cross encoder models.

    Cross encoders score the similarity between pairs of text. They are
    commonly used for reranking retrieved documents based on relevance
    to a query.

    Implementations should override the `score` method to compute
    similarity scores for pairs of texts.
    """

    @abstractmethod
    def score(self, text_pairs: list[tuple[str, str]]) -> list[float]:
        """Score the similarity of text pairs.

        Args:
            text_pairs: A list of tuples, where each tuple contains
                two strings to compare. Typically (query, document) pairs.

        Returns:
            A list of float scores where higher values indicate greater
            similarity. The length of the returned list matches the
            length of text_pairs.

        Example:
            .. code-block:: python

                scores = encoder.score([
                    ("machine learning", "ML is a subset of AI"),
                    ("machine learning", "cooking recipes"),
                ])
                # scores might be [0.95, 0.12]
        """
