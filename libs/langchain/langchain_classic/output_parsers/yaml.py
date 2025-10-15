import json
import re
from typing import TypeVar

import yaml
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser
from pydantic import BaseModel, ValidationError
from typing_extensions import override

from langchain_classic.output_parsers.format_instructions import (
    YAML_FORMAT_INSTRUCTIONS,
)

T = TypeVar("T", bound=BaseModel)


class YamlOutputParser(BaseOutputParser[T]):
    """Parse YAML output using a Pydantic model."""

    pydantic_object: type[T]
    """The Pydantic model to parse."""
    pattern: re.Pattern = re.compile(
        r"^```(?:ya?ml)?(?P<yaml>[^`]*)",
        re.MULTILINE | re.DOTALL,
    )
    """Regex pattern to match yaml code blocks
    within triple backticks with optional yaml or yml prefix."""

    @override
    def parse(self, text: str) -> T:
        try:
            # Greedy search for 1st yaml candidate.
            match = re.search(self.pattern, text.strip())
            # If no backticks were present, try to parse the entire output as yaml.
            yaml_str = match.group("yaml") if match else text

            json_object = yaml.safe_load(yaml_str)
            return self.pydantic_object.model_validate(json_object)

        except (yaml.YAMLError, ValidationError) as e:
            name = self.pydantic_object.__name__
            msg = f"Failed to parse {name} from completion {text}. Got: {e}"
            raise OutputParserException(msg, llm_output=text) from e

    @override
    def get_format_instructions(self) -> str:
        # Copy schema to avoid altering original Pydantic schema.
        schema = dict(self.pydantic_object.model_json_schema().items())

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
    @override
    def OutputType(self) -> type[T]:
        return self.pydantic_object
