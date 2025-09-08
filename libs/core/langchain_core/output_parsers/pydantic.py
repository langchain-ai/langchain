"""Output parsers using Pydantic."""

import json
from typing import Annotated, Generic, Optional

import pydantic
from pydantic import SkipValidation
from typing_extensions import override

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.outputs import Generation
from langchain_core.utils.pydantic import (
    PydanticBaseModel,
    TBaseModel,
)


class PydanticOutputParser(JsonOutputParser, Generic[TBaseModel]):
    """Parse an output using a pydantic model."""

    pydantic_object: Annotated[type[TBaseModel], SkipValidation()]
    """The pydantic model to parse."""

    def _parse_obj(self, obj: dict) -> TBaseModel:
        try:
            if issubclass(self.pydantic_object, pydantic.BaseModel):
                return self.pydantic_object.model_validate(obj)
            if issubclass(self.pydantic_object, pydantic.v1.BaseModel):
                return self.pydantic_object.parse_obj(obj)
            msg = f"Unsupported model version for PydanticOutputParser: \
                        {self.pydantic_object.__class__}"
            raise OutputParserException(msg)
        except (pydantic.ValidationError, pydantic.v1.ValidationError) as e:
            raise self._parser_exception(e, obj) from e

    def _parser_exception(
        self, e: Exception, json_object: dict
    ) -> OutputParserException:
        json_string = json.dumps(json_object)
        name = self.pydantic_object.__name__
        msg = f"Failed to parse {name} from completion {json_string}. Got: {e}"
        return OutputParserException(msg, llm_output=json_string)

    def parse_result(
        self, result: list[Generation], *, partial: bool = False
    ) -> Optional[TBaseModel]:
        """Parse the result of an LLM call to a pydantic object.

        Args:
            result: The result of the LLM call.
            partial: Whether to parse partial JSON objects.
                If True, the output will be a JSON object containing
                all the keys that have been returned so far.
                Defaults to False.

        Raises:
            OutputParserException: If the result is not valid JSON
                or does not conform to the pydantic model.

        Returns:
            The parsed pydantic object.
        """
        try:
            json_object = super().parse_result(result)
            return self._parse_obj(json_object)
        except OutputParserException:
            if partial:
                return None
            raise

    def parse(self, text: str) -> TBaseModel:
        """Parse the output of an LLM call to a pydantic object.

        Args:
            text: The output of the LLM call.

        Returns:
            The parsed pydantic object.
        """
        return super().parse(text)

    def get_format_instructions(self) -> str:
        """Return the format instructions for the JSON output.

        Returns:
            The format instructions for the JSON output.
        """
        # Copy schema to avoid altering original Pydantic schema.
        schema = dict(self.pydantic_object.model_json_schema().items())

        # Remove extraneous fields.
        reduced_schema = schema
        if "title" in reduced_schema:
            del reduced_schema["title"]
        if "type" in reduced_schema:
            del reduced_schema["type"]
        # Ensure json in context is well-formed with double quotes.
        schema_str = json.dumps(reduced_schema, ensure_ascii=False)

        return _PYDANTIC_FORMAT_INSTRUCTIONS.format(schema=schema_str)

    @property
    def _type(self) -> str:
        return "pydantic"

    @property
    @override
    def OutputType(self) -> type[TBaseModel]:
        """Return the pydantic model."""
        return self.pydantic_object


_PYDANTIC_FORMAT_INSTRUCTIONS = """The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

Here is the output schema:
```
{schema}
```"""  # noqa: E501

# Re-exporting types for backwards compatibility
__all__ = [
    "PydanticBaseModel",
    "PydanticOutputParser",
    "TBaseModel",
]
