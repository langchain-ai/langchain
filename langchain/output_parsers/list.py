from __future__ import annotations

from abc import abstractmethod
from typing import List

from langchain.output_parsers.base import BaseOutputParser


class ListOutputParser(BaseOutputParser):
    """Class to parse the output of an LLM call to a list."""

    @abstractmethod
    def parse(self, text: str) -> List[str]:
        """Parse the output of an LLM call."""


class CommaSeparatedListOutputParser(ListOutputParser):
    """Parse out comma separated lists."""

    def parse(self, text: str) -> List[str]:
        """Parse the output of an LLM call."""
        return text.strip().split(", ")
