from __future__ import annotations

import json
from typing import Any, List

from pydantic import BaseModel

from langchain.output_parsers.format_instructions import STRUCTURED_FORMAT_INSTRUCTIONS
from langchain.schema import BaseOutputParser, OutputParserException

line_template = '\t"{name}": {type}  // {description}'


class ResponseSchema(BaseModel):
    name: str
    description: str


def _get_sub_string(schema: ResponseSchema) -> str:
    return line_template.format(
        name=schema.name, description=schema.description, type="string"
    )


def parse_json_markdown(text: str, expected_keys: List[str]) -> Any:
    if "```json" not in text:
        raise OutputParserException(
            f"Got invalid return object. Expected markdown code snippet with JSON "
            f"object, but got:\n{text}"
        )

    json_string = text.split("```json")[1].strip().strip("```").strip()
    try:
        json_obj = json.loads(json_string)
    except json.JSONDecodeError as e:
        raise OutputParserException(f"Got invalid JSON object. Error: {e}")
    for key in expected_keys:
        if key not in json_obj:
            raise OutputParserException(
                f"Got invalid return object. Expected key `{key}` "
                f"to be present, but got {json_obj}"
            )
    return json_obj


class StructuredOutputParser(BaseOutputParser):
    response_schemas: List[ResponseSchema]

    @classmethod
    def from_response_schemas(
        cls, response_schemas: List[ResponseSchema]
    ) -> StructuredOutputParser:
        return cls(response_schemas=response_schemas)

    def get_format_instructions(self) -> str:
        schema_str = "\n".join(
            [_get_sub_string(schema) for schema in self.response_schemas]
        )
        return STRUCTURED_FORMAT_INSTRUCTIONS.format(format=schema_str)

    def parse(self, text: str) -> Any:
        expected_keys = [rs.name for rs in self.response_schemas]
        return parse_json_markdown(text, expected_keys)

    @property
    def _type(self) -> str:
        return "structured"
