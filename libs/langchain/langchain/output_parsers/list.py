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
        return (
            "Ответ должен быть списком, разделенным запятыми, "
            "например: `foo, bar, baz`. Кроме списка в ответе \
не должно быть никаких других слов."
        )

    def parse(self, text: str) -> List[str]:
        """Parse the output of an LLM call."""
        return text.strip().split(", ")
