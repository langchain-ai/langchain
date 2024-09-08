from __future__ import annotations

import re
from typing import Any

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser


class ReflectionOutputParser(BaseOutputParser[str]):
    """Parse the output of an LLM call to extract content within <output> tags."""

    @property
    def _type(self) -> str:
        return "reflection_output_parser"

    def parse(self, text: str) -> str:
        """Extract content within <output> tags from the input text.

        Args:
            text: The input text to parse.

        Returns:
            The extracted content within <output> tags.

        Raises:
            OutputParserException: If no <output> tags are found in the text.
        """
        pattern = r'<output>(.*?)</output>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        raise OutputParserException("No <output> tags found in the text.")

    def get_format_instructions(self) -> str:
        """Return instructions on how to format the output."""
        return (
            "Your response should be enclosed within <output> tags. "
            "For example: <output>Your response here</output>"
        )
