"""Parser for ECharts output."""

from __future__ import annotations

from typing_extensions import override

from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.utils.json import parse_json_markdown


class EChartsOutputParser(BaseOutputParser[dict]):
    """Explore the LLM output and retrieve a JSON dictionary for ECharts.

    This parser expects the LLM to provide a JSON object that matches the
    Apache ECharts 'option' configuration.

    !!! warning
        This output parser is experimental and may change in future versions.
    """

    @override
    def parse(self, text: str) -> dict:
        """Parse the output of an LLM call to a JSON object.

        Args:
            text: The output of the LLM call.

        Returns:
            The parsed JSON object as a dictionary.
        """
        result = parse_json_markdown(text)
        if isinstance(result, list):
            if len(result) > 0 and isinstance(result[0], dict):
                return result[0]
            return {}
        return result if isinstance(result, dict) else {}

    @override
    def get_format_instructions(self) -> str:
        """Return the format instructions for the ECharts JSON output.

        Returns:
            The format instructions for the ECharts JSON output.
        """
        return (
            "The output should be a valid JSON object following the Apache ECharts "
            "'option' schema. For example:\n"
            '```json\n{\n  "title": { "text": "ECharts Example" },\n'
            '  "xAxis": { ... },\n  "yAxis": { ... },\n  "series": [ ... ]\n}\n```'
        )

    @property
    @override
    def _type(self) -> str:
        """Return the output parser type for serialization."""
        return "echarts_output_parser"


__all__ = ["EChartsOutputParser"]
