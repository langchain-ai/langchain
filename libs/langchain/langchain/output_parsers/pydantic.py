import json
import re
from typing import Type, TypeVar

from langchain.output_parsers.format_instructions import PYDANTIC_FORMAT_INSTRUCTIONS
from langchain.pydantic_v1 import BaseModel, ValidationError
from langchain.schema import BaseOutputParser, OutputParserException

T = TypeVar("T", bound=BaseModel)


class PydanticOutputParser(BaseOutputParser[T]):
    """Parse an output using a pydantic model."""

    pydantic_object: Type[T]
    """The pydantic model to parse."""

    def parse(self, text: str) -> T:
        try:
            # Greedy search for 1st json candidate.
            match = re.search(
                r"\{.*\}", text.strip(), re.MULTILINE | re.IGNORECASE | re.DOTALL
            )
            json_str = ""
            if match:
                json_str = match.group()
            json_object = json.loads(json_str, strict=False)
            return self.pydantic_object.parse_obj(json_object)

        except (json.JSONDecodeError, ValidationError) as e:
            name = self.pydantic_object.__name__
            msg = f"Failed to parse {name} from completion {text}. Got: {e}"
            raise OutputParserException(msg, llm_output=text)

    def get_format_instructions(self) -> str:
        schema = self.pydantic_object.schema()

        # Remove extraneous fields.
        reduced_schema = schema
        if "title" in reduced_schema:
            del reduced_schema["title"]
        if "type" in reduced_schema:
            del reduced_schema["type"]
        # Ensure json in context is well-formed with double quotes.
        schema_str = json.dumps(reduced_schema)

        return PYDANTIC_FORMAT_INSTRUCTIONS.format(schema=schema_str)

    @property
    def _type(self) -> str:
        return "pydantic"
