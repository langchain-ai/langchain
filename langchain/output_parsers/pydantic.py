import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError

from langchain.output_parsers.format_instructions import PYDANTIC_FORMAT_INSTRUCTIONS
from langchain.schema import BaseOutputParser, OutputParserException


class PydanticOutputParser(BaseOutputParser):
    pydantic_object: Any

    def parse(self, text: str) -> BaseModel:
        try:
            # Greedy search for 1st json candidate.
            match = re.search(
                "\{.*\}", text.strip(), re.MULTILINE | re.IGNORECASE | re.DOTALL
            )
            json_str = ""
            if match:
                json_str = match.group()
            json_object = json.loads(json_str)
            return self.pydantic_object.parse_obj(json_object)

        except (json.JSONDecodeError, ValidationError) as e:
            name = self.pydantic_object.__name__
            msg = f"Failed to parse {name} from completion {text}. Got: {e}"
            raise OutputParserException(msg)

    def get_format_instructions(self) -> str:
        schema = self.pydantic_object.schema()

        # Remove extraneous fields.
        reduced_schema = schema
        if "title" in reduced_schema:
            del reduced_schema["title"]
        if "type" in reduced_schema:
            del reduced_schema["type"]
        # Ensure json in context is well-formed with double quotes.
        schema = json.dumps(reduced_schema)

        return PYDANTIC_FORMAT_INSTRUCTIONS.format(schema=schema)
