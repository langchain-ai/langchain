import json
from typing import Any, List, Type

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.outputs import Generation
from langchain_core.pydantic_v1 import BaseModel, ValidationError

from langchain.output_parsers.format_instructions import PYDANTIC_FORMAT_INSTRUCTIONS


class PydanticOutputParser(JsonOutputParser):
    """Parse an output using a pydantic model."""

    pydantic_object: Type[BaseModel]
    """The pydantic model to parse.
    
    Attention: To avoid potential compatibility issues, it's recommended to use
        pydantic <2 or leverage the v1 namespace in pydantic >= 2.
    """

    def parse_result(self, result: List[Generation], *, partial: bool = False) -> Any:
        json_object = super().parse_result(result)
        try:
            return self.pydantic_object.parse_obj(json_object)
        except ValidationError as e:
            name = self.pydantic_object.__name__
            msg = f"Failed to parse {name} from completion {json_object}. Got: {e}"
            raise OutputParserException(msg, llm_output=json_object)

    def get_format_instructions(self) -> str:
        # Copy schema to avoid altering original Pydantic schema.
        schema = {k: v for k, v in self.pydantic_object.schema().items()}

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

    @property
    def OutputType(self) -> Type[BaseModel]:
        """Return the pydantic model."""
        return self.pydantic_object
