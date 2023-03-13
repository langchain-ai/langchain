import json
import subprocess
from typing import Type

from pydantic import BaseModel
from tempfile import TemporaryDirectory
from langchain.output_parsers.base import BaseOutputParser
import logging

logger = logging.getLogger(__name__)

format_instructions = """The output should be a markdown code snippet formatted in the following schema:

```typescript
{format}
```"""
line_template = """"{name}": {type}"""

MAX_DEPTH = 6


def parse_json_schema_to_typescript(schema: dict) -> str:
    """Return the string representation of schema."""
    with TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "schema.json"), "w") as f:
            json.dump(schema, f)
        schema_str = subprocess.check_output(
            ["json2ts", "-i", "schema.json"], cwd=tmpdir
        ).decode("utf-8")
    return schema_str


class PydanticOutputParser(BaseOutputParser):
    response_schema: Type[BaseModel]

    def get_format_instructions(self) -> str:
        schema = self.response_schema.schema()
        schema_str = get_nested_schema_str(schema)
        return format_instructions.format(format=schema_str)

    def parse(self, text: str) -> BaseModel:
        json_string = text.split("```typescript")[1].strip().strip("```").strip()
        json_obj = json.loads(json_string)
        return self.response_schema(**json_obj)
