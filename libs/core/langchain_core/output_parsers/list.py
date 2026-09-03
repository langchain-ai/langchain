import csv
import io
import re
from abc import abstractmethod
from typing import Any, ClassVar, TypeVar

from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.outputs import Generation

T = TypeVar("T")


class ListOutputParser(BaseOutputParser[list[str]]):
    """Base class for list output parsers."""

    @abstractmethod
    def parse(self, text: str) -> list[str]:
        """Parse the output of an LLM call."""

    def parse_result(
        self, result: list[Generation], *, partial: bool = False, **kwargs: Any
    ) -> list[str]:
        """Parse a list of candidate generations into outputs."""
        return self.parse(result[0].text)


class CommaSeparatedListOutputParser(ListOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""

    def get_format_instructions(self) -> str:
        return (
            "Your response should be a list of comma separated values, "
            "eg: foo, bar, baz"
        )

    def parse(self, text: str) -> list[str]:
        """Parse the output of an LLM call."""
        try:
            reader = csv.reader(io.StringIO(text), skipinitialspace=True)
            row = next(reader)
            return [item.strip() for item in row if item.strip()]
        except Exception:
            return [item.strip() for item in text.split(",") if item.strip()]

    @property
    def _type(self) -> str:
        return "comma-separated-list"


def parse_markdown_list(text: str) -> list[str]:
    """Parse a markdown list."""
    lines = text.split("\n")
    items = []
    for line in lines:
        match = re.match(MarkdownListOutputParser.pattern, line)
        if match:
            items.append(match.group(2))
    return items


class NumberedListOutputParser(ListOutputParser):
    """Parse a numbered list."""

    pattern: ClassVar[str] = r"^\s*(\d+)\.\s([^\n]+)$"
    """The pattern to match a numbered list item."""

    def get_format_instructions(self) -> str:
        return (
            "Your response should be a numbered list with each item on a new line. "
            "For example: \n\n1. foo\n\n2. bar\n\n3. baz"
        )

    def parse(self, text: str) -> list[str]:
        """Parse the output of an LLM call."""
        lines = text.split("\n")
        items = []
        for line in lines:
            match = re.match(self.pattern, line)
            if match:
                items.append(match.group(2))
        return items

    @property
    def _type(self) -> str:
        return "numbered-list"


class MarkdownListOutputParser(ListOutputParser):
    """Parse a markdown list (supports -, *, and + bullet points)."""

    pattern: ClassVar[str] = r"^\s*([-*+])\s([^\n]+)$"
    """The pattern to match a markdown list item."""

    def get_format_instructions(self) -> str:
        return (
            "Your response should be a markdown list with each item on a new line. "
            "For example: \n- foo\n- bar\n- baz"
        )

    def parse(self, text: str) -> list[str]:
        """Parse the output of an LLM call."""
        return parse_markdown_list(text)

    @property
    def _type(self) -> str:
        return "markdown-list"
