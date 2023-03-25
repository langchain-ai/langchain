from __future__ import annotations

import json
from typing import List

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

    def parse(self, text: str) -> BaseModel:
        json_string = text.split("```json")[1].strip().strip("```").strip()
        json_obj = json.loads(json_string)
        for schema in self.response_schemas:
            if schema.name not in json_obj:
                raise OutputParserException(
                    f"Got invalid return object. Expected key `{schema.name}` "
                    f"to be present, but got {json_obj}"
                )
        return json_obj
