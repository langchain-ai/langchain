from __future__ import annotations

import re
from typing import Dict, Optional

from langchain.schema import BaseOutputParser


class RegexDictParser(BaseOutputParser):
    """Parse the output of an LLM call into a Dictionary using a regex."""

    regex_pattern: str = r"{}:\s?([^.'\n']*)\.?"  # : :meta private:
    """The regex pattern to use to parse the output."""
    output_key_to_format: Dict[str, str]
    """The keys to use for the output."""
    no_update_value: Optional[str] = None
    """The default key to use for the output."""

    @property
    def _type(self) -> str:
        """Return the type key."""
        return "regex_dict_parser"

    def parse(self, text: str) -> Dict[str, str]:
        """Parse the output of an LLM call."""
        result = {}
        for output_key, expected_format in self.output_key_to_format.items():
            specific_regex = self.regex_pattern.format(re.escape(expected_format))
            matches = re.findall(specific_regex, text)
            if not matches:
                raise ValueError(
                    f"No match found for output key: {output_key} with expected format \
                        {expected_format} on text {text}"
                )
            elif len(matches) > 1:
                raise ValueError(
                    f"Multiple matches found for output key: {output_key} with \
                        expected format {expected_format} on text {text}"
                )
            elif (
                self.no_update_value is not None and matches[0] == self.no_update_value
            ):
                continue
            else:
                result[output_key] = matches[0]
        return result
