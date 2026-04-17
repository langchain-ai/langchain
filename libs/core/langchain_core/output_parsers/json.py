"""Parser for JSON output."""

from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Annotated, Any, TypeVar

import jsonpatch  # type: ignore[import-untyped]
import pydantic
from pydantic import SkipValidation
from pydantic.v1 import BaseModel
from typing_extensions import override

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers.format_instructions import JSON_FORMAT_INSTRUCTIONS
from langchain_core.output_parsers.transform import BaseCumulativeTransformOutputParser
from langchain_core.outputs import Generation
from langchain_core.utils.json import (
    parse_and_check_json_markdown,
    parse_json_markdown,
    parse_partial_json,
)

# Union type needs to be last assignment to PydanticBaseModel to make mypy happy.
PydanticBaseModel = BaseModel | pydantic.BaseModel

TBaseModel = TypeVar("TBaseModel", bound=PydanticBaseModel)


class JsonOutputParser(BaseCumulativeTransformOutputParser[Any]):
    """Parse the output of an LLM call to a JSON object.

    Probably the most reliable output parser for getting structured data that does *not*
    use function calling.

    When used in streaming mode, it will yield partial JSON objects containing all the
    keys that have been returned so far.

    In streaming, if `diff` is set to `True`, yields `JSONPatch` operations describing
    the difference between the previous and the current object.
    """

    pydantic_object: Annotated[type[TBaseModel] | None, SkipValidation()] = None  # type: ignore[valid-type]
    """The Pydantic object to use for validation.

    If `None`, no validation is performed.
    """

    @override
    def _diff(self, prev: Any | None, next: Any) -> Any:
        """Compute JSONPatch operations between parsed outputs.

        Args:
            prev: The previously parsed JSON-compatible object.
            next: The current parsed JSON-compatible object.

        Returns:
            A list of JSONPatch operations describing the change from `prev` to `next`.
        """
        return jsonpatch.make_patch(prev, next).patch

    @staticmethod
    def _get_schema(pydantic_object: type[TBaseModel]) -> dict[str, Any]:
        """Get a JSON schema from a Pydantic model class.

        Uses `model_json_schema()` for Pydantic v2 models and `schema()` for
        Pydantic v1 models.

        Args:
            pydantic_object: The Pydantic model class to introspect.

        Returns:
            The JSON schema dictionary for the model.
        """
        if issubclass(pydantic_object, pydantic.BaseModel):
            return pydantic_object.model_json_schema()
        return pydantic_object.schema()

    @override
    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """Parse the result of an LLM call to a JSON object.

        Args:
            result: The result of the LLM call.
            partial: Whether to parse partial JSON objects.

                If `True`, the output will be a JSON object containing all the keys that
                have been returned so far.

                If `False`, the output will be the full JSON object.

        Returns:
            The parsed JSON object.

        Raises:
            OutputParserException: If the output is not valid JSON.
        """
        text = result[0].text
        text = text.strip()
        if partial:
            try:
                return parse_json_markdown(text)
            except JSONDecodeError:
                return None
        else:
            try:
                return parse_json_markdown(text)
            except JSONDecodeError as e:
                msg = f"Invalid json output: {text}"
                raise OutputParserException(msg, llm_output=text) from e

    def parse(self, text: str) -> Any:
        """Parse LLM output text to a JSON object.

        Wraps `parse_result` by converting the input text into a single
        `Generation` instance.

        Args:
            text: LLM output text.

        Returns:
            The parsed JSON object.

        Raises:
            OutputParserException: If the output is not valid JSON.
        """
        return self.parse_result([Generation(text=text)])

    def get_format_instructions(self) -> str:
        """Return format instructions for JSON output.

        When `pydantic_object` is `None`, returns a minimal instruction to
        produce a JSON object. When a Pydantic model is provided, returns
        schema-constrained JSON instructions derived from that model.

        Returns:
            A string describing the expected JSON output format.
        """
        if self.pydantic_object is None:
            return "Return a JSON object."
        # Copy schema to avoid altering original Pydantic schema.
        schema = dict(self._get_schema(self.pydantic_object).items())

        # Remove extraneous fields.
        reduced_schema = schema
        if "title" in reduced_schema:
            del reduced_schema["title"]
        if "type" in reduced_schema:
            del reduced_schema["type"]
        # Ensure json in context is well-formed with double quotes.
        schema_str = json.dumps(reduced_schema, ensure_ascii=False)
        return JSON_FORMAT_INSTRUCTIONS.format(schema=schema_str)

    @property
    def _type(self) -> str:
        return "simple_json_output_parser"


# For backwards compatibility
SimpleJsonOutputParser = JsonOutputParser


__all__ = [
    "JsonOutputParser",
    "SimpleJsonOutputParser",  # For backwards compatibility
    "parse_and_check_json_markdown",  # For backwards compatibility
    "parse_partial_json",  # For backwards compatibility
]
