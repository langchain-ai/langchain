import json
import re
import warnings
from typing import Type, TypeVar, Optional, List, Any

from pydantic import BaseModel, ValidationError, create_model, Field

from langchain.output_parsers.format_instructions import PYDANTIC_FORMAT_INSTRUCTIONS
from langchain.schema import BaseOutputParser, OutputParserException

T = TypeVar("T", bound=BaseModel)


class PydanticOutputParser(BaseOutputParser[T]):
    """Generates an output instruction and complimentary parser for a Pydantic model, and any nested models within it.

    Args:
        pydantic_object (Type[T]): The Pydantic model to parse the output into.
        excluded_fields (Optional[List[str]], optional): A list of fields to exclude from the Pydantic model. Defaults to None.
    """

    pydantic_object: Type[T]
    excluded_fields: Optional[List[str]] = None
    dynamic_model: Optional[Type[T]] = None

    @property
    def _type(self) -> str:
        return "pydantic"

    def parse(self, text: str) -> Any:
        try:
            # Greedy search for 1st json candidate.
            json_str = ""
            if match := re.search(r"\{.*\}", text.strip(), re.MULTILINE | re.IGNORECASE | re.DOTALL):
                json_str = match.group()

            json_object = json.loads(json_str, strict=False)

            if self.excluded_fields is not None:
                json_object = self._handle_excluded_fields(json_object)

            model_to_use = self.dynamic_model or self.pydantic_object
            return model_to_use.parse_obj(json_object)

        except (json.JSONDecodeError, ValidationError) as e:
            name = (self.dynamic_model or self.pydantic_object).__name__
            msg = f"Failed to parse {name} from completion {text}. Got: {e}"
            raise OutputParserException(msg) from e

    def get_format_instructions(self) -> str:
        """Get the format instructions for the Pydantic model."""
        schema = self.pydantic_object.schema()

        if self.excluded_fields is not None:
            schema = self._filter_schema(schema)

        # Remove extraneous fields: 'title' and 'type'
        reduced_schema = {key: value for key, value in schema.items() if key not in ["title", "type"]}

        # Ensure json in context is well-formed with double quotes.
        schema_str = json.dumps(reduced_schema)
        return PYDANTIC_FORMAT_INSTRUCTIONS.format(schema=schema_str)

    def _handle_excluded_fields(self, json_object: dict) -> dict:
        """Return a new json object with the excluded fields removed.
        If required fields are excluded, create a dynamic Pydantic model and assign it to self.dynamic_model.
        """
        json_object = self._filter_excluded_fields_dict(json_object)

        # Retrieve required fields and collect any that are excluded
        required_fields = self.pydantic_object.schema().get("required", [])
        if self.excluded_fields is not None and (
            excluded_required_fields := [field for field in self.excluded_fields if field in required_fields]
        ):
            # Raise a warning that required fields are being excluded
            warnings.warn(f"The following required fields are being excluded: {', '.join(excluded_required_fields)}")

            # Dynamically create a new Pydantic model with only the fields that are not excluded
            field_defaults = {
                name: field.default for name, field in self.pydantic_object.__fields__.items() if not field.required
            }
            fields = {
                name: (field.outer_type_, Field(field_defaults.get(name, ...)))
                for name, field in self.pydantic_object.__fields__.items()
                if name not in self.excluded_fields
            }

            self.dynamic_model = create_model(self.pydantic_object.__name__, **fields)  # type: ignore[call-overload]

            # Check if json_object can be parsed by DynamicModel
            try:
                if issubclass(self.dynamic_model, BaseModel):
                    self.dynamic_model.parse_obj(json_object)
            except ValidationError as e:
                model_name = (self.dynamic_model or self.pydantic_object).__name__
                msg = f"Failed to load JSON for {model_name} from json object: {json_object}. Got: {e}"
                raise OutputParserException(msg) from e

        return json_object

    def exclude_fields(self, fields: List[str]) -> None:
        """Update the parser to exclude specified fields."""
        self.excluded_fields = fields

    def _filter_properties_and_required(self, data: dict) -> dict:
        """Filter 'properties' and 'required' fields from a dictionary to exclude the specified fields."""
        if "properties" in data:
            data["properties"] = self._filter_excluded_fields_dict(data["properties"])
        if "required" in data:
            data["required"] = self._filter_excluded_fields_list(data["required"])
        return data

    def _filter_excluded_fields_dict(self, data: dict) -> dict:
        """Exclude specified fields from a dictionary."""
        if isinstance(data, dict) and self.excluded_fields is not None:
            return {key: value for key, value in data.items() if key not in self.excluded_fields}
        else:
            return data

    def _filter_excluded_fields_list(self, data: list) -> list:
        """Exclude specified fields from a list."""
        if isinstance(data, list) and self.excluded_fields is not None:
            return [item for item in data if item not in self.excluded_fields]
        else:
            return data

    def _filter_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Filter the schema to exclude the specified fields."""
        filtered_schema = schema.copy()

        # Remove excluded fields from the 'properties' dictionary and the 'required' list
        filtered_schema = self._filter_properties_and_required(filtered_schema)

        # Remove excluded fields and their definitions from the nested objects' schemas
        definitions = filtered_schema.get("definitions", {})
        filtered_definitions = {}
        for definition_name, definition in definitions.items():
            filtered_definition = definition.copy()
            filtered_definition = self._filter_properties_and_required(filtered_definition)

            # Only keep this definition if it has any properties left after filtering
            if filtered_definition["properties"]:
                filtered_definitions[definition_name] = filtered_definition

        filtered_schema["definitions"] = filtered_definitions

        # Remove definitions and required lists if they are empty
        filtered_schema = {
            key: value for key, value in filtered_schema.items() if value and key not in ["title", "type"]
        }

        return filtered_schema
