from __future__ import annotations

import re
from typing import Dict, List, Optional

from langchain.schema import BaseOutputParser


class RegexParser(BaseOutputParser):
    """Class to parse the output into a dictionary."""

    regex: str
    output_keys: List[str]
    default_output_key: Optional[str] = None

    @property
    def _type(self) -> str:
        """Return the type key."""
        return "regex_parser"

    def parse(self, text: str) -> Dict[str, str]:
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
