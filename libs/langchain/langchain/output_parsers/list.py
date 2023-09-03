from __future__ import annotations

import re
from abc import abstractmethod
from typing import List

from langchain.schema import BaseOutputParser


class ListOutputParser(BaseOutputParser[List[str]]):
    """Parse the output of an LLM call to a list."""

    @property
    def _type(self) -> str:
        return "list"

    @abstractmethod
    def parse(self, text: str) -> List[str]:
        """Parse the output of an LLM call."""


class CommaSeparatedListOutputParser(ListOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""

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

    @property
    def _type(self) -> str:
        return "comma-separated-list"


class NumberedListOutputParser(ListOutputParser):
    """Parse a numbered list."""

    def get_format_instructions(self) -> str:
        return (
            "Your response should be a numbered list with each item on a new line. "
            "For example: \n\n1. foo\n\n2. bar\n\n3. baz"
        )

    def parse(self, text: str) -> List[str]:
        """Parse the output of an LLM call."""
        pattern = r"\d+\.\s([^\n]+)"

        # Extract the text of each item
        matches = re.findall(pattern, text)
        return matches

    @property
    def _type(self) -> str:
        return "numbered-list"
