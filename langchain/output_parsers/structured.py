import json
from typing import List

from pydantic import BaseModel

from langchain.output_parsers.base import BaseOutputParser

format_instructions = """The output should be a markdown code snippet formatted in the following schema:

```json
{{
{format}
}}
```"""
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
    def from_response_schemas(cls, response_schemas: List[ResponseSchema]):
        return cls(response_schemas=response_schemas)

    def get_format_instructions(self) -> str:
        schema_str = "\n".join(
            [_get_sub_string(schema) for schema in self.response_schemas]
        )
        return format_instructions.format(format=schema_str)

    def parse(self, text: str) -> BaseModel:
        json_string = text.split("```json")[1].strip().strip("```").strip()
        json_obj = json.loads(json_string)
        return json_obj
