from __future__ import annotations

import re
from typing import Any, TypeVar

from langchain.schema import BaseOutputParser, OutputParserException, PromptValue

T = TypeVar("T")


class ChainedOutputParser(BaseOutputParser[T]):
    """Chains multiple output parsers together."""

    parsers: list[BaseOutputParser]

    def parse(self, text: str) -> T:
        """Parse the output of an LLM call by chaining multiple parsers together."""
        value = text
        for parser in self.parsers:
            value = parser.parse(value)
        return value

    def parse_with_prompt(self, completion: str, prompt: PromptValue) -> Any:
        """Parse the output of an LLM call with a given prompt by chaining multiple parsers together."""
        value = completion
        for parser in self.parsers:
            value = parser.parse_with_prompt(value, prompt)
        return value

    def get_format_instructions(self) -> str:
        """Get the formatting instructions for the first parser in the chain."""
        return self.parsers[0].get_format_instructions()
