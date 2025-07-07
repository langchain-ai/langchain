from __future__ import annotations

import re
from typing import Optional

from langchain_core.output_parsers import BaseOutputParser


class RegexDictParser(BaseOutputParser[dict[str, str]]):
    """Parse the output of an LLM call into a Dictionary using a regex."""

    regex_pattern: str = r"{}:\s?([^.'\n']*)\.?"  # : :meta private:
    """The regex pattern to use to parse the output."""
    output_key_to_format: dict[str, str]
    """The keys to use for the output."""
    no_update_value: Optional[str] = None
    """The default key to use for the output."""

    @property
    def _type(self) -> str:
        """Return the type key."""
        return "regex_dict_parser"

    def parse(self, text: str) -> dict[str, str]:
        """Parse the output of an LLM call."""
        result = {}
        for output_key, expected_format in self.output_key_to_format.items():
            specific_regex = self.regex_pattern.format(re.escape(expected_format))
            matches = re.findall(specific_regex, text)
            if not matches:
                msg = (
                    f"No match found for output key: {output_key} with expected format \
                        {expected_format} on text {text}"
                )
                raise ValueError(msg)
            if len(matches) > 1:
                msg = f"Multiple matches found for output key: {output_key} with \
                        expected format {expected_format} on text {text}"
                raise ValueError(msg)
            if self.no_update_value is not None and matches[0] == self.no_update_value:
                continue
            result[output_key] = matches[0]
        return result
