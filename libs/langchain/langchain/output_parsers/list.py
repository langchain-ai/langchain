from __future__ import annotations

from abc import abstractmethod
from typing import List

from langchain.schema import BaseOutputParser


class ListOutputParser(BaseOutputParser):
    """Class to parse the output of an LLM call to a list."""

    @property
    def _type(self) -> str:
        return "list"

    @abstractmethod
    def parse(self, text: str) -> List[str]:
        """Parse the output of an LLM call."""


class CommaSeparatedListOutputParser(ListOutputParser):
    """Parse out comma separated lists."""

    @property
    def lc_serializable(self) -> bool:
        return True

    def get_format_instructions(self) -> str:
        return (
            "Your response should be a list of comma separated values, "
            "eg: `foo, bar, baz`"
        )

    def parse(self, text: str) -> List[str]:
        """Parse the output of an LLM call."""
        return text.strip().split(", ")
