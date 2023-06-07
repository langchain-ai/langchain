from __future__ import annotations

from typing import Any, List

from pydantic import BaseModel

from langchain.output_parsers.format_instructions import (
    STRUCTURED_FORMAT_INSTRUCTIONS,
    STRUCTURED_INPUT_FORMAT_INSTRUCTIONS
)
from langchain.output_parsers.json import parse_and_check_json_markdown
from langchain.schema import BaseOutputParser

line_template = '\t"{name}": {type}  // {description}'


class ResponseSchema(BaseModel):
    name: str
    description: str
    type: str = "string"


def _get_sub_string(schema: ResponseSchema) -> str:
    return line_template.format(
        name=schema.name, description=schema.description, type=schema.type
    )


class StructuredOutputParser(BaseOutputParser):
    response_schemas: List[ResponseSchema]

    @classmethod
    def from_response_schemas(
        cls, response_schemas: List[ResponseSchema]
    ) -> StructuredOutputParser:
        return cls(response_schemas=response_schemas)

    def get_format_instructions(self, input_format=False) -> str:
        """
        Method to get the format instructions for the output parser.

        Args:
            input_format (bool): Whether to get the format instructions for the input or output. Defaults to False (output format instructions)

        example:
        ```python
        from langchain.output_parsers.structured import StructuredOutputParser, ResponseSchema

        response_schemas = [
            ResponseSchema(name="foo", description="a list of strings", type="List[string]", items={"type": "string"}),
            ResponseSchema(name="bar", description="a string", type="string"),
        ]

        parser = StructuredOutputParser.from_response_schemas(response_schemas)

        print(parser.get_format_instructions())

        output:
        # The output should be a markdown code snippet formatted in the following schema, including the leading and trailing "```json" and "```":
        #
        # ```json
        # {
        #     "foo": List[string]  // a list of strings
        #     "bar": string  // a string
        # }

        NOTE: if the input_format is True, only the first row change with the following:
        # The input will be a markdown code snippet formatted in the following schema, including the leading and trailing "```json" and "```":
        """
        schema_str = "\n".join(
            [_get_sub_string(schema) for schema in self.response_schemas]
        )
        return STRUCTURED_FORMAT_INSTRUCTIONS.format(format=schema_str)


    def parse(self, text: str) -> Any:
        expected_keys = [rs.name for rs in self.response_schemas]
        return parse_and_check_json_markdown(text, expected_keys)

    @property
    def _type(self) -> str:
        return "structured"
