"""Fake parsers for testing purposes."""


from typing import Dict, Mapping, Optional

from langchain.prompts.base import DictOutputParser


class FakeDictParser(DictOutputParser):
    """Fake dict parser for testing purposes."""

    parses: Optional[Mapping] = None

    def parse(self, text: str) -> Dict[str, str]:
        """Return fake parse from input."""
        if self.parses is not None:
            return self.parses[text]
        return {
            "tool": "Fake tool",
            "tool_input": "Fake input",
        }
