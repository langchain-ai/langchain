from __future__ import annotations

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
        return " ответ напиши через запятую одной строкой!"

    def parse(self, text: str) -> List[str]:
        """Parse the output of an LLM call."""
        if ", " not in text and "\n" in text:
            text = text.replace("\n", ", ")
        return text.strip().split(", ")
