import json
from typing import Type

from pydantic import BaseModel

from langchain.output_parsers.base import BaseOutputParser

format_instructions = """The output should be a markdown code snippet formatted in the following schema:

```typescript
{{
{format}
}}
```"""
line_template = """\t"{name}": {type}  // {description}"""


def _get_sub_string(k: str, v: dict) -> str:
    return line_template.format(name=k, description=v["description"], type=v["type"])


class PydanticOutputParser(BaseOutputParser):
    response_schema: Type[BaseModel]

    def get_format_instructions(self) -> str:
        schema_str = "\n".join(
            [
                _get_sub_string(k, v)
                for k, v in self.response_schema.schema()["properties"].items()
            ]
        )
        return format_instructions.format(format=schema_str)

    def parse(self, text: str) -> BaseModel:
        json_string = text.split("```typescript")[1].strip().strip("```").strip()
        json_obj = json.loads(json_string)
        return self.response_schema(**json_obj)
