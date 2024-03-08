import json
from typing import Generic, List, Type, TypeVar

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.output_parsers.pydantic import _PYDANTIC_FORMAT_INSTRUCTIONS
from langchain_core.outputs.generation import Generation
from langchain_core.pydantic_v2 import BaseModel, ValidationError

TBaseModel = TypeVar("TBaseModel", bound=BaseModel)


class PydanticV2OutputParser(JsonOutputParser, Generic[TBaseModel]):
    """Parse an output using a pydantic model."""

    pydantic_v2_object: Type[TBaseModel]
    """The pydantic model to parse."""

    def parse_result(
        self, result: List[Generation], *, partial: bool = False
    ) -> TBaseModel:
        json_object = super().parse_result(result)
        try:
            return self.pydantic_v2_object.model_validate(json_object)
        except ValidationError as e:
            name = self.pydantic_v2_object.__name__
            msg = f"Failed to parse {name} from completion {json_object}. Got: {e}"
            raise OutputParserException(msg, llm_output=json_object)

    def parse(self, text: str) -> TBaseModel:
        return super().parse(text)

    def get_format_instructions(self) -> str:
        # Copy schema to avoid altering original Pydantic schema.
        schema = {k: v for k, v in self.pydantic_v2_object.model_json_schema().items()}

        # Remove extraneous fields.
        reduced_schema = schema
        if "title" in reduced_schema:
            del reduced_schema["title"]
        if "type" in reduced_schema:
            del reduced_schema["type"]
        # Ensure json in context is well-formed with double quotes.
        schema_str = json.dumps(reduced_schema)

        return _PYDANTIC_FORMAT_INSTRUCTIONS.format(schema=schema_str)

    @property
    def _type(self) -> str:
        return "pydantic"

    @property
    def OutputType(self) -> Type[TBaseModel]:
        """Return the pydantic model."""
        return self.pydantic_v2_object
