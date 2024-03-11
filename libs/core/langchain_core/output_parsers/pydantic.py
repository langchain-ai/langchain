import json
from typing import Generic, List, Type, TypeVar, Union

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.outputs import Generation
from langchain_core.pydantic_v1 import BaseModel, ValidationError
from langchain_core.pydantic_v2 import BaseModel as V2BaseModel
from langchain_core.pydantic_v2 import ValidationError as V2ValidationError
from langchain_core.utils.pydantic import PYDANTIC_MAJOR_VERSION

PydanticBaseModel = BaseModel
if PYDANTIC_MAJOR_VERSION == 2:
    PydanticBaseModel = Union[BaseModel, V2BaseModel]  # type: ignore

TBaseModel = TypeVar("TBaseModel", bound=PydanticBaseModel)


class PydanticOutputParser(JsonOutputParser, Generic[TBaseModel]):
    """Parse an output using a pydantic model."""

    pydantic_object: Type[TBaseModel]
    """The pydantic model to parse."""

    def _parse_obj(self, obj: dict) -> TBaseModel:
        if PYDANTIC_MAJOR_VERSION == 2 and issubclass(
            self.pydantic_object, V2BaseModel
        ):
            try:
                return self.pydantic_object.model_validate(obj)
            except V2ValidationError as e:
                raise self._parser_exception(str(e), json.dumps(obj))
        elif issubclass(self.pydantic_object, BaseModel):
            return self.pydantic_object.parse_obj(obj)
        else:
            raise OutputParserException(
                f"Unsupported model version for PydanticOutputParser: \
                    {self.pydantic_object.__class__}"
            )

    def _parser_exception(self, e: str, json_object: str) -> OutputParserException:
        name = self.pydantic_object.__name__
        msg = f"Failed to parse {name} from completion {json_object}. Got: {e}"
        return OutputParserException(msg, llm_output=json_object)

    def parse_result(
        self, result: List[Generation], *, partial: bool = False
    ) -> TBaseModel:
        json_object = super().parse_result(result)
        try:
            return self._parse_obj(json_object)
        except ValidationError as e:
            raise self._parser_exception(str(e), json_object)

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
