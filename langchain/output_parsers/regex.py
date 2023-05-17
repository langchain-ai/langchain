from __future__ import annotations

import re
from typing import Callable, Dict, Generic, List, Optional, TypeVar

from langchain.schema import BaseOutputParser, OutputParserException

_PARSED_T = TypeVar("_PARSED_T")


class ListRegexParser(BaseOutputParser[List[_PARSED_T]], Generic[_PARSED_T]):
    """Class to parse output using a regex."""

    regex: str
    _cast: Callable[[str], _PARSED_T]

    def parse(self, text: str) -> List[_PARSED_T]:
        """Parse the output of an LLM call."""
        matches = re.findall(self.regex, text)
        if matches:
            return [self._cast(m) for m in matches]
        else:
            raise OutputParserException(f"Could not parse output: {text}")

    @property
    def _type(self) -> str:
        """Return the type key."""
        return "list_regex_parser"


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
                raise OutputParserException(f"Could not parse output: {text}")
            else:
                return {
                    key: text if key == self.default_output_key else ""
                    for key in self.output_keys
                }
