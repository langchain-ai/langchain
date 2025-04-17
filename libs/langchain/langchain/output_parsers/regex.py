from __future__ import annotations

import re
from typing import Optional

from langchain_core.output_parsers import BaseOutputParser


class RegexParser(BaseOutputParser[dict[str, str]]):
    """Parse the output of an LLM call using a regex."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    regex: str
    """The regex to use to parse the output."""
    output_keys: list[str]
    """The keys to use for the output."""
    default_output_key: Optional[str] = None
    """The default key to use for the output."""

    @property
    def _type(self) -> str:
        """Return the type key."""
        return "regex_parser"

    def parse(self, text: str) -> dict[str, str]:
        """Parse the output of an LLM call."""
        match = re.search(self.regex, text)
        if match:
            return {key: match.group(i + 1) for i, key in enumerate(self.output_keys)}
        else:
            if self.default_output_key is None:
                raise ValueError(f"Could not parse output: {text}")
            else:
                return {
                    key: text if key == self.default_output_key else ""
                    for key in self.output_keys
                }
