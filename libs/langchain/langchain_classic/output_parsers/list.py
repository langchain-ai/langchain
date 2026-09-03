import re
from collections.abc import Iterator

from langchain_core.output_parsers.base import BaseOutputParser
from typing_extensions import override


class ListOutputParser(BaseOutputParser[list[str]]):
    """Class to parse the output of an LLM call to a list."""

    def parse(self, text: str) -> list[str]:
        """Parse the output of an LLM call."""
        raise NotImplementedError


class CommaSeparatedListOutputParser(ListOutputParser):
    """Parse a comma-separated list."""

    @override
    def get_format_instructions(self) -> str:
        return (
            "Your response should be a list of comma separated values, "
            "e.g.: foo, bar, baz"
        )

    def parse(self, text: str) -> list[str]:
        """Parse the output of an LLM call."""
        return [item.strip() for item in text.split(",")]

    @property
    def _type(self) -> str:
        return "comma-separated-list"


class MarkdownListOutputParser(ListOutputParser):
    """Parse a markdown list (supports -, *, and + bullet points)."""

    pattern: str = r"^\s*([-*+])\s([^\n]+)$"
    """The pattern to match a markdown list item."""

    @override
    def get_format_instructions(self) -> str:
        return (
            "Your response should be a markdown list with each item on a new line. "
            "For example: \n\n- foo\n- bar\n- baz"
        )

    def parse(self, text: str) -> list[str]:
        """Parse the output of an LLM call.

        Args:
            text: The output of an LLM call.

        Returns:
            A list of strings.
        """
        matches = re.findall(self.pattern, text, re.MULTILINE)
        # re.findall with a single capturing group in pattern returns the group;
        # here we have two groups (bullet and text), so we extract the second element.
        return [match[1] for match in matches]

    @override
    def parse_iter(self, text: str) -> Iterator[re.Match[str]]:
        return re.finditer(self.pattern, text, re.MULTILINE)

    @property
    def _type(self) -> str:
        return "markdown-list"


class NumberedListOutputParser(ListOutputParser):
    """Parse a numbered list."""

    pattern: str = r"^\s*(\d+)\.\s([^\n]+)$"
    """The pattern to match a numbered list item."""

    @override
    def get_format_instructions(self) -> str:
        return (
            "Your response should be a numbered list with each item on a new line. "
            "For example: \n\n1. foo\n\n2. bar\n\n3. baz"
        )

    def parse(self, text: str) -> list[str]:
        """Parse the output of an LLM call.

        Args:
            text: The output of an LLM call.

        Returns:
            A list of strings.
        """
        matches = re.findall(self.pattern, text, re.MULTILINE)
        return [match[1] for match in matches]

    @override
    def parse_iter(self, text: str) -> Iterator[re.Match[str]]:
        return re.finditer(self.pattern, text, re.MULTILINE)

    @property
    def _type(self) -> str:
        return "numbered-list"


__all__ = [
    "CommaSeparatedListOutputParser",
    "ListOutputParser",
    "MarkdownListOutputParser",
    "NumberedListOutputParser",
]
