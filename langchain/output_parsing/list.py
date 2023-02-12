"""Parse out comma separated lists."""
from typing import List

from langchain.output_parsing.base import ListOutputParser


class CommaSeparatedListOutputParser(ListOutputParser):
    """Parse out comma separated lists."""

    def parse(self, text: str) -> List[str]:
        """Parse the output of an LLM call."""
        return text.strip().split(", ")
