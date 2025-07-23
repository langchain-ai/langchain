from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import BaseOutputParser
from langchain_core.utils import pre_init

_MIN_PARSERS = 2


class CombiningOutputParser(BaseOutputParser[dict[str, Any]]):
    """Combine multiple output parsers into one."""

    parsers: list[BaseOutputParser]

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @pre_init
    def validate_parsers(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Validate the parsers."""
        parsers = values["parsers"]
        if len(parsers) < _MIN_PARSERS:
            msg = "Must have at least two parsers"
            raise ValueError(msg)
        for parser in parsers:
            if parser._type == "combining":
                msg = "Cannot nest combining parsers"
                raise ValueError(msg)
            if parser._type == "list":
                msg = "Cannot combine list parsers"
                raise ValueError(msg)
        return values

    @property
    def _type(self) -> str:
        """Return the type key."""
        return "combining"

    def get_format_instructions(self) -> str:
        """Instructions on how the LLM output should be formatted."""

        initial = f"For your first output: {self.parsers[0].get_format_instructions()}"
        subsequent = "\n".join(
            f"Complete that output fully. Then produce another output, separated by two newline characters: {p.get_format_instructions()}"  # noqa: E501
            for p in self.parsers[1:]
        )
        return f"{initial}\n{subsequent}"

    def parse(self, text: str) -> dict[str, Any]:
        """Parse the output of an LLM call."""
        texts = text.split("\n\n")
        output = {}
        for txt, parser in zip(texts, self.parsers):
            output.update(parser.parse(txt.strip()))
        return output
