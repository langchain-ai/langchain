import json
import re
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

from langchain.output_parsers.format_instructions import PYDANTIC_FORMAT_INSTRUCTIONS
from langchain.schema import BaseOutputParser, OutputParserException

T = TypeVar("T", bound=BaseModel)


class PydanticOutputParser(BaseOutputParser[T]):
    pydantic_object: Type[T]

    def parse(self, text: str) -> T:
        try:
            text = self._remove_illegal_quatations(text)
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
        schema_str = json.dumps(reduced_schema)

        return PYDANTIC_FORMAT_INSTRUCTIONS.format(schema=schema_str)

    def _remove_illegal_quatations(self, text: str) -> str:
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if "[" in line and "]" in line:
                continue

            matches = re.finditer(r'"', line)
            indices = [match.start() for match in matches]

            if len(indices) < 4:
                continue

            if indices:
                first_quote, second_quote = indices[0], indices[1]
                third_quote, last_quote = indices[2], indices[-1]
                # Replace illegal " to '
                handle_line = line[third_quote + 1 : last_quote].replace('"', "'")
                lines[i] = (
                    line[:first_quote]
                    + line[first_quote:second_quote]
                    + line[second_quote : third_quote + 1]
                    + handle_line
                    + line[last_quote:]
                )

        return "\n".join(lines)

    @property
    def _type(self) -> str:
        return "pydantic"
