from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import BaseOutputParser
from langchain_core.output_parsers.json import parse_and_check_json_markdown
from pydantic import BaseModel

from langchain.output_parsers.format_instructions import (
    STRUCTURED_FORMAT_INSTRUCTIONS,
    STRUCTURED_FORMAT_SIMPLE_INSTRUCTIONS,
)

line_template = '\t"{name}": {type}  // {description}'


class ResponseSchema(BaseModel):
    """Schema for a response from a structured output parser."""

    name: str
    """The name of the schema."""
    description: str
    """The description of the schema."""
    type: str = "string"
    """The type of the response."""


def _get_sub_string(schema: ResponseSchema) -> str:
    return line_template.format(
        name=schema.name, description=schema.description, type=schema.type
    )


class StructuredOutputParser(BaseOutputParser[dict[str, Any]]):
    """Parse the output of an LLM call to a structured output."""

    response_schemas: list[ResponseSchema]
    """The schemas for the response."""

    @classmethod
    def from_response_schemas(
        cls, response_schemas: list[ResponseSchema]
    ) -> StructuredOutputParser:
        return cls(response_schemas=response_schemas)

    def get_format_instructions(self, only_json: bool = False) -> str:
        """Get format instructions for the output parser.

        example:
        ```python
        from langchain.output_parsers.structured import (
            StructuredOutputParser, ResponseSchema
        )

        response_schemas = [
            ResponseSchema(
                name="foo",
                description="a list of strings",
                type="List[string]"
                ),
            ResponseSchema(
                name="bar",
                description="a string",
                type="string"
                ),
        ]

        parser = StructuredOutputParser.from_response_schemas(response_schemas)

        print(parser.get_format_instructions())  # noqa: T201

        output:
        # The output should be a Markdown code snippet formatted in the following
        # schema, including the leading and trailing "```json" and "```":
        #
        # ```json
        # {
        #     "foo": List[string]  // a list of strings
        #     "bar": string  // a string
        # }
        # ```

        Args:
            only_json (bool): If True, only the json in the Markdown code snippet
                will be returned, without the introducing text. Defaults to False.
        """
        schema_str = "\n".join(
            [_get_sub_string(schema) for schema in self.response_schemas]
        )
        if only_json:
            return STRUCTURED_FORMAT_SIMPLE_INSTRUCTIONS.format(format=schema_str)
        else:
            return STRUCTURED_FORMAT_INSTRUCTIONS.format(format=schema_str)

    def parse(self, text: str) -> dict[str, Any]:
        expected_keys = [rs.name for rs in self.response_schemas]
        return parse_and_check_json_markdown(text, expected_keys)

    @property
    def _type(self) -> str:
        return "structured"
