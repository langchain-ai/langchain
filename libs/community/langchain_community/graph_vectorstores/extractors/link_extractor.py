from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Iterable, Set, TypeVar

from langchain_core.graph_vectorstores import Link

InputT = TypeVar("InputT")

METADATA_LINKS_KEY = "links"


class LinkExtractor(ABC, Generic[InputT]):
    """Interface for extracting links (incoming, outgoing, bidirectional)."""

    @abstractmethod
    def extract_one(self, input: InputT) -> set[Link]:  # noqa: A002
        """Add edges from each `input` to the corresponding documents.

        Args:
            input: The input content to extract edges from.

        Returns:
            Set of links extracted from the input.
        """

    def extract_many(self, inputs: Iterable[InputT]) -> Iterable[Set[Link]]:
        """Add edges from each `input` to the corresponding documents.

        Args:
            inputs: The input content to extract edges from.

        Returns:
            Iterable over the set of links extracted from the input.
        """
        return map(self.extract_one, inputs)
