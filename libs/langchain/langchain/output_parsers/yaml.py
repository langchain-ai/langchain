import json
import re
from typing import TypeVar

import yaml
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser
from pydantic import BaseModel, ValidationError

from langchain.output_parsers.format_instructions import YAML_FORMAT_INSTRUCTIONS

T = TypeVar("T", bound=BaseModel)


class YamlOutputParser(BaseOutputParser[T]):
    """Parse YAML output using a pydantic model."""

    pydantic_object: type[T]
    """The pydantic model to parse."""
    pattern: re.Pattern = re.compile(
        r"^```(?:ya?ml)?(?P<yaml>[^`]*)", re.MULTILINE | re.DOTALL
    )
    """Regex pattern to match yaml code blocks 
    within triple backticks with optional yaml or yml prefix."""

    def parse(self, text: str) -> T:
        try:
            # Greedy search for 1st yaml candidate.
            match = re.search(self.pattern, text.strip())
            yaml_str = ""
            if match:
                yaml_str = match.group("yaml")
            else:
                # If no backticks were present, try to parse the entire output as yaml.
                yaml_str = text

            json_object = yaml.safe_load(yaml_str)
            if hasattr(self.pydantic_object, "model_validate"):
                return self.pydantic_object.model_validate(json_object)
            else:
                return self.pydantic_object.parse_obj(json_object)

        except (yaml.YAMLError, ValidationError) as e:
            name = self.pydantic_object.__name__
            msg = f"Failed to parse {name} from completion {text}. Got: {e}"
            raise OutputParserException(msg, llm_output=text) from e

    def get_format_instructions(self) -> str:
        # Copy schema to avoid altering original Pydantic schema.
        schema = {k: v for k, v in self.pydantic_object.schema().items()}

        # Remove extraneous fields.
        reduced_schema = schema
        if "title" in reduced_schema:
            del reduced_schema["title"]
        if "type" in reduced_schema:
            del reduced_schema["type"]
        # Ensure yaml in context is well-formed with double quotes.
        schema_str = json.dumps(reduced_schema)

        return YAML_FORMAT_INSTRUCTIONS.format(schema=schema_str)

    @property
    def _type(self) -> str:
        return "yaml"

    @property
    def OutputType(self) -> type[T]:
        return self.pydantic_object
