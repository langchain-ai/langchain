"""Parser for ECharts output."""

from __future__ import annotations

import json
from typing import Any, List

from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.outputs import Generation
from langchain_core.utils.json import parse_json_markdown


class EChartsOutputParser(BaseOutputParser[dict]):
    """Explore the LLM output and retrieve a JSON dictionary for ECharts.

    This parser expects the LLM to provide a JSON object that matches the 
    Apache ECharts 'option' configuration.
    """

    def parse(self, text: str) -> dict:
        """Parse the output of an LLM call to a JSON object.

        Args:
            text: The output of the LLM call.

        Returns:
            The parsed JSON object as a dictionary.
        """
        return parse_json_markdown(text)

    def get_format_instructions(self) -> str:
        """Return the format instructions for the ECharts JSON output.

        Returns:
            The format instructions for the ECharts JSON output.
        """
        return (
            "The output should be a valid JSON object following the Apache ECharts "
            "'option' schema. For example:\n"
            '```json\n{\n  "title": { "text": "ECharts Example" },\n  "xAxis": { ... },\n  "yAxis": { ... },\n  "series": [ ... ]\n}\n```'
        )

    @property
    def _type(self) -> str:
        return "echarts_output_parser"


__all__ = ["EChartsOutputParser"]
