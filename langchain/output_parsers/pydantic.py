import json
import re
import warnings
from typing import Type, TypeVar, Optional, List, Any, Union

from pydantic import BaseModel, ValidationError, create_model, Field

from langchain.output_parsers.format_instructions import PYDANTIC_FORMAT_INSTRUCTIONS
from langchain.schema import BaseOutputParser, OutputParserException

T = TypeVar("T", bound=BaseModel)


class PydanticOutputParser(BaseOutputParser[T]):
    """Generates an output instruction for a Pydantic model, and any nested models within it.
    . And parse the output into a Pydantic model.
        
    Args:
        pydantic_object (Type[T]): The Pydantic model to parse the output into.
        excluded_fields (Optional[List[str]], optional): A list of fields to exclude from the Pydantic model. Defaults to None."""
    pydantic_object: Type[T]
    excluded_fields: Optional[List[str]] = None  # Define an instance variable to hold the excluded fields

    @property
    def _type(self) -> str:
        return "pydantic"

    def parse(self, text: str) -> Any:
        try:
            # Greedy search for 1st json candidate.
            match = re.search(
                r"\{.*\}", text.strip(), re.MULTILINE | re.IGNORECASE | re.DOTALL
            )
            json_str = ""
            if match:
                json_str = match.group()
                
            json_object = json.loads(json_str, strict=False)
            
            if self.excluded_fields is not None:
                json_object = self._handle_excluded_fields(json_object)
            
            return self.pydantic_object.parse_obj(json_object)

        except (json.JSONDecodeError, ValidationError) as e:
            name = self.pydantic_object.__name__
            msg = f"Failed to parse {name} from completion {text}. Got: {e}"
            raise OutputParserException(msg)

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

    def _handle_excluded_fields(self, json_object):
        """Return a new json object with the excluded fields removed.
        If required fields are excluded, return a new dynamically created Pydantic model with the excluded fields removed."""
        json_object = self._filter_excluded_fields(json_object)

        # Check if any of the excluded fields are required
        required_fields = self.pydantic_object.schema().get('required', [])
        excluded_required_fields = [field for field in self.excluded_fields if field in required_fields]

        if excluded_required_fields:
            # Raise a warning if any required fields are excluded
            warnings.warn(f"The following required fields are being excluded: {', '.join(excluded_required_fields)}")

            # Dynamically create a new Pydantic model with only the fields that are not excluded
            field_defaults = {name: field.default for name, field in self.pydantic_object.__fields__.items() if not field.required}
            fields = {name: (field.outer_type_, Field(field_defaults.get(name, ...))) for name, field in self.pydantic_object.__fields__.items() if name not in self.excluded_fields}
            DynamicModel = create_model(self.pydantic_object.__name__, **fields)

            return DynamicModel.parse_obj(json_object)
        return json_object

    def exclude_fields(self, fields: List[str]) -> None:
        """Update the parser to exclude specified fields."""
        self.excluded_fields = fields

    def _filter_properties_and_required(self, data: dict) -> dict:
        """Filter 'properties' and 'required' fields from a dictionary to exclude the specified fields."""
        if 'properties' in data:
            data['properties'] = self._filter_excluded_fields(data['properties'])
        if 'required' in data:
            data['required'] = self._filter_excluded_fields(data['required'])
        return data

    def _filter_excluded_fields(self, data: Union[dict, list]) -> Union[dict, list]:
        """Exclude specified fields from a dictionary or a list."""
        if isinstance(data, dict):
            return {key: value for key, value in data.items() if key not in self.excluded_fields}
        elif isinstance(data, list):
            return [item for item in data if item not in self.excluded_fields]
        

    def _filter_schema(self, schema: dict) -> dict:
        """Filter the schema to exclude the specified fields."""
        filtered_schema = schema.copy()

        # Remove excluded fields from the 'properties' dictionary and the 'required' list
        filtered_schema = self._filter_properties_and_required(filtered_schema)

        # Remove excluded fields and their definitions from the nested objects' schemas
        definitions = filtered_schema.get('definitions', {})
        filtered_definitions = {}
        for definition_name, definition in definitions.items():
            filtered_definition = definition.copy()
            filtered_definition = self._filter_properties_and_required(filtered_definition)

            # Only keep this definition if it has any properties left after filtering
            if filtered_definition['properties']:
                filtered_definitions[definition_name] = filtered_definition

        filtered_schema['definitions'] = filtered_definitions
        
        # Remove definitions and required lists if they are empty
        filtered_schema = {key: value for key, value in filtered_schema.items() if value and key not in ["title", "type"]}

        return filtered_schema
