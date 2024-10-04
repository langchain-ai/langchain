from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Annotated, Any, Optional, TypeVar, Union

import jsonpatch  # type: ignore[import]
import pydantic
from pydantic import SkipValidation

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers.format_instructions import JSON_FORMAT_INSTRUCTIONS
from langchain_core.output_parsers.transform import BaseCumulativeTransformOutputParser
from langchain_core.outputs import Generation
from langchain_core.utils.json import (
    parse_and_check_json_markdown,
    parse_json_markdown,
    parse_partial_json,
)
from langchain_core.utils.pydantic import PYDANTIC_MAJOR_VERSION

if PYDANTIC_MAJOR_VERSION < 2:
    PydanticBaseModel = pydantic.BaseModel

else:
    from pydantic.v1 import BaseModel

    # Union type needs to be last assignment to PydanticBaseModel to make mypy happy.
    PydanticBaseModel = Union[BaseModel, pydantic.BaseModel]  # type: ignore

TBaseModel = TypeVar("TBaseModel", bound=PydanticBaseModel)


class JsonOutputParser(BaseCumulativeTransformOutputParser[Any]):
    """Parse the output of an LLM call to a JSON object.

    When used in streaming mode, it will yield partial JSON objects containing
    all the keys that have been returned so far.

    In streaming, if `diff` is set to `True`, yields JSONPatch operations
    describing the difference between the previous and the current object.
    """

    pydantic_object: Annotated[Optional[type[TBaseModel]], SkipValidation()] = None  # type: ignore
    """The Pydantic object to use for validation.
    If None, no validation is performed."""

    def _diff(self, prev: Optional[Any], next: Any) -> Any:
        return jsonpatch.make_patch(prev, next).patch

    def _get_schema(self, pydantic_object: type[TBaseModel]) -> dict[str, Any]:
        if issubclass(pydantic_object, pydantic.BaseModel):
            return pydantic_object.model_json_schema()
        elif issubclass(pydantic_object, pydantic.v1.BaseModel):
            return pydantic_object.schema()

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """Parse the result of an LLM call to a JSON object.

        Args:
            result: The result of the LLM call.
            partial: Whether to parse partial JSON objects.
                If True, the output will be a JSON object containing
                all the keys that have been returned so far.
                If False, the output will be the full JSON object.
                Default is False.

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
        """Parse the output of an LLM call to a JSON object.

        Args:
            text: The output of the LLM call.

        Returns:
            The parsed JSON object.
        """
        return self.parse_result([Generation(text=text)])

    def get_format_instructions(self) -> str:
        """Return the format instructions for the JSON output.

        Returns:
            The format instructions for the JSON output.
        """
        if self.pydantic_object is None:
            return "Return a JSON object."
        else:
            # Copy schema to avoid altering original Pydantic schema.
            schema = dict(self._get_schema(self.pydantic_object).items())

            # Remove extraneous fields.
            reduced_schema = schema
            if "title" in reduced_schema:
                del reduced_schema["title"]
            if "type" in reduced_schema:
                del reduced_schema["type"]
            # Ensure json in context is well-formed with double quotes.
            schema_str = json.dumps(reduced_schema)
            return JSON_FORMAT_INSTRUCTIONS.format(schema=schema_str)

    @property
    def _type(self) -> str:
        return "simple_json_output_parser"


# For backwards compatibility
SimpleJsonOutputParser = JsonOutputParser
parse_partial_json = parse_partial_json
parse_and_check_json_markdown = parse_and_check_json_markdown
