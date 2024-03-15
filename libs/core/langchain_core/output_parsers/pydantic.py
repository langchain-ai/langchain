import json
from typing import Generic, List, Type, TypeVar

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.outputs import Generation
from langchain_core.pydantic_v1 import BaseModel, ValidationError

TBaseModel = TypeVar("TBaseModel", bound=BaseModel)


class PydanticOutputParser(JsonOutputParser, Generic[TBaseModel]):
    """Parse an output using a pydantic model."""

    pydantic_object: Type[TBaseModel]
    """The pydantic model to parse.
    
    Attention: To avoid potential compatibility issues, it's recommended to use
        pydantic <2 or leverage the v1 namespace in pydantic >= 2.
    """

    def parse_result(
        self, result: List[Generation], *, partial: bool = False
    ) -> TBaseModel:
        json_object = super().parse_result(result)
        try:
            return self.pydantic_object.parse_obj(json_object)
        except ValidationError as e:
            name = self.pydantic_object.__name__
            msg = f"Failed to parse {name} from completion {json_object}. Got: {e}"
            raise OutputParserException(msg, llm_output=json_object)

    def parse(self, text: str) -> TBaseModel:
        return super().parse(text)

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

        return _PYDANTIC_FORMAT_INSTRUCTIONS.format(schema=schema_str)

    @property
    def _type(self) -> str:
        return "pydantic"

    @property
    def OutputType(self) -> Type[TBaseModel]:
        """Return the pydantic model."""
        return self.pydantic_object


_PYDANTIC_FORMAT_INSTRUCTIONS = """The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

Here is the output schema:
```
{schema}
```"""  # noqa: E501
