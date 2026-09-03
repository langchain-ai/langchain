"""Hierarchical robust JSON output parser for LangChain."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel, Field

try:
    from langchain_core.exceptions import OutputParserException
    from langchain_core.output_parsers import BaseOutputParser
except ImportError:
    class OutputParserException(ValueError):  # type: ignore[no-redef]
        """Exception raised when an LLM generation cannot be parsed into structured format."""
        pass

    class BaseOutputParser(BaseModel):  # type: ignore[no-redef]
        """Base output parser fallback."""
        pass

T = TypeVar("T")


class HierarchicalRobustJsonParser(BaseOutputParser):
    """Output parser that extracts, repairs, and parses nested JSON from LLM generations.

    Handles common LLM edge cases:
    - Markdown fenced code blocks (```json ... ```, ``` ... ```)
    - Single-quoted string values
    - Trailing commas before closing brackets / braces
    - Unbalanced brackets from partial or truncated responses
    - Unquoted keys
    - Optional Pydantic schema validation

    Example:
        .. code-block:: python

            from langchain_community.output_parsers import HierarchicalRobustJsonParser

            parser = HierarchicalRobustJsonParser()
            result = parser.parse("Here is the JSON:\n```json\n{'name': 'Alice', 'age': 30,}\n```")
    """

    pydantic_object: Optional[Type[BaseModel]] = Field(
        default=None,
        description="Optional Pydantic model for schema validation.",
    )
    strict_validation: bool = Field(
        default=False,
        description="Whether to fail strictly when Pydantic validation encounters errors.",
    )

    class Config:
        arbitrary_types_allowed = True

    @property
    def _type(self) -> str:
        return "hierarchical_robust_json_parser"

    def _extract_json_substring(self, text: str) -> str:
        """Extract JSON substring from markdown code blocks or surrounding text."""
        code_block_match = re.search(
            r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE
        )
        if code_block_match:
            return code_block_match.group(1).strip()

        json_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
        if json_match:
            return json_match.group(1).strip()

        return text.strip()

    def _repair_json_string(self, json_str: str) -> str:
        """Apply heuristics to repair malformed JSON from LLMs."""
        s = json_str.strip()

        # Remove single-line JS comments
        s = re.sub(r"//.*$", "", s, flags=re.MULTILINE)

        # Convert single-quoted values to double-quoted values
        s = re.sub(r"(?<=[{,:\s])'([^']+)'(?=[\s:,}\]])", r'"\1"', s)

        # Quote unquoted keys
        s = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)', r'\1"\2"\3', s)

        # Remove trailing commas
        s = re.sub(r",\s*([\]}])", r"\1", s)

        # Balance unclosed brackets/braces
        stack: List[str] = []
        in_string = False
        escape = False

        for char in s:
            if char == '"' and not escape:
                in_string = not in_string
            elif not in_string:
                if char in "{[":
                    stack.append("}" if char == "{" else "]")
                elif char in "}]":
                    if stack and stack[-1] == char:
                        stack.pop()
            escape = char == "\\" and not escape

        while stack:
            s += stack.pop()

        return s

    def parse(self, text: str) -> Any:
        """Parse LLM generation into structured JSON or validated Pydantic model.

        Args:
            text: Raw string output from an LLM.

        Returns:
            Parsed dictionary, list, or Pydantic instance.

        Raises:
            OutputParserException: If JSON cannot be parsed or repaired.
        """
        raw_candidate = self._extract_json_substring(text)

        # Attempt 1: Direct standard JSON decode
        try:
            parsed = json.loads(raw_candidate)
            return self._validate(parsed)
        except json.JSONDecodeError:
            pass

        # Attempt 2: Repaired JSON decode
        repaired_candidate = self._repair_json_string(raw_candidate)
        try:
            parsed = json.loads(repaired_candidate)
            return self._validate(parsed)
        except json.JSONDecodeError as err:
            raise OutputParserException(
                f"Failed to parse or repair JSON from LLM output: {text}. "
                f"Error: {str(err)}"
            ) from err

    async def aparse(self, text: str) -> Any:
        """Asynchronously parse LLM generation into structured output."""
        return self.parse(text)

    def _validate(self, parsed: Any) -> Any:
        """Validate parsed structure against Pydantic schema if provided."""
        if self.pydantic_object is not None:
            if isinstance(parsed, dict) and hasattr(
                self.pydantic_object, "model_validate"
            ):
                try:
                    return self.pydantic_object.model_validate(parsed)
                except Exception as e:
                    if self.strict_validation:
                        raise OutputParserException(
                            f"Pydantic schema validation failed: {str(e)}"
                        ) from e
            elif isinstance(parsed, list) and not self.strict_validation:
                return parsed

        return parsed

    def get_format_instructions(self) -> str:
        """Return instructions for the LLM to format its output as JSON."""
        if self.pydantic_object and hasattr(
            self.pydantic_object, "model_json_schema"
        ):
            schema = json.dumps(self.pydantic_object.model_json_schema(), indent=2)
            return (
                f"Return a valid JSON object matching the following schema:\n"
                f"```json\n{schema}\n```"
            )
        return "Return a valid JSON object or array."
