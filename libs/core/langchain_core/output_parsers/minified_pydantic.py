"""Minification tool class for Output parsers using Pydantic."""

from typing import List, Type, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field

from langchain_core.output_parsers.pydantic import PydanticOutputParser


class MinifiedPydanticOutputParser(PydanticOutputParser):
    """
    This tool class is used to minify a Pydantic schema by replacing the original
    field names with short identifiers ('a', 'b', 'c', ...), while **preserving all field descriptions**.

    It also provides a way to reverse the transformation: given a response using the minified keys,
    it restores the original schema with full field names and descriptions.

    Use case:
    - Reduce payload size in API requests/responses.
    - Maintain field-level documentation during minification.

    Example:
    --------
    Original schema:
        class User(BaseModel):
            first_name: str = Field(..., description="The user's first name")
            last_name: str = Field(..., description="The user's last name")
            email: str = Field(..., description="The user's email address")

    After minification:
        class UserMinified(BaseModel):
            a: str = Field(..., description="The user's first name")
            b: str = Field(..., description="The user's last name")
            c: str = Field(..., description="The user's email address")

    When a response is received using minified keys, the tool can reverse it back to the
    original schema.
    """

    model_config = ConfigDict(extra="allow")

    def __init__(self, pydantic_object: Type[BaseModel]):
        """
        Initialize the Minificator with the original Pydantic model type.

        Args:
            original_type (Type[BaseModel]): The original Pydantic model type to be minified.
        """
        super().__init__(pydantic_object=pydantic_object)
        self.field_names_mapper = {}
        self.original_type = pydantic_object
        self.small_names_list = self._get_small_name_list()
        self.minified = self._make_fields_required_and_small(self.original_type)
        self.pydantic_object = self.minified

    def parse_result(self, result: list[dict], *, partial: bool = False) -> BaseModel:
        """
        Parse the result of an LLM call to a Pydantic object, using the minified schema.

        Args:
            result (list[dict]): The result of the LLM call.
            partial (bool): Whether to parse partial JSON objects.

        Returns:
            BaseModel: An instance of the original Pydantic model with full field names.
        """
        parsed = super().parse_result(result, partial=partial)
        return self.get_original(parsed)

    def _make_fields_required_and_small(
        self, pydantic_cls: Type[BaseModel]
    ) -> Type[BaseModel]:
        # Get the fields from the Pydantic class via __fields__
        original_fields = pydantic_cls.model_fields

        # Create a dictionary to store the new field annotations and field info
        new_annotations = {}
        new_fields = {}

        for name, field in original_fields.items():
            short_name = ""

            # Use field.annotation to get the type
            field_type = field.annotation

            # MANDATORY FIRST CHANGE Remove Optional if it's part of the field type (Union)
            if get_origin(field_type) is Union:
                # Filter out `NoneType` from the Union
                field_type = next(
                    t for t in get_args(field_type) if t is not type(None)
                )

            # If the field type is a subclass of BaseModel, recursively apply the function
            if isinstance(field_type, type) and issubclass(field_type, BaseModel):
                short_name = self._get_short_field_name(name)
                field_type = self._make_fields_required_and_small(
                    field_type
                )  # Recursively transform BaseModel fields

            if get_origin(field_type) is list or get_origin(field_type) is List:
                list_inner_type = get_args(field_type)[0]
                if get_origin(list_inner_type) is Union:
                    list_inner_type = next(
                        t for t in get_args(list_inner_type) if t is not type(None)
                    )
                if issubclass(list_inner_type, BaseModel):
                    short_name = self._get_short_field_name(name)
                    inner_field_type = self._make_fields_required_and_small(
                        list_inner_type
                    )
                    field_type = List[inner_field_type]

            # Add the field as required (no default values)
            if short_name == "":
                short_name = self._get_short_field_name(name)
            new_annotations[short_name] = field_type

            new_fields[short_name] = Field(
                ...,
                alias=field.alias,
                description=field.description,
                serialization_alias=name,
            )

        # Dynamically create a new class with proper annotations and fields
        new_class_name = f"Minified{pydantic_cls.__name__}"

        # Define the class dictionary including annotations
        class_dict = {"__annotations__": new_annotations}
        class_dict.update(new_fields)  # Update the dictionary with fields

        # Create the new class
        new_class = type(new_class_name, (BaseModel,), class_dict)

        return new_class

    def _remove_none_values(self, data: Union[dict, list, any]) -> Union[dict, list]:
        if isinstance(data, dict):
            # Recursively clean each key-value pair
            return {
                k: self._remove_none_values(v) for k, v in data.items() if v is not None
            }

        if isinstance(data, list):
            # Recursively clean each item in the list
            return [self._remove_none_values(item) for item in data if item is not None]

        # Return non-iterable values as they are
        return data

    def get_original(self, llm_result: Union[BaseModel, dict]) -> BaseModel:
        if not isinstance(llm_result, BaseModel):
            llm_result = self._remove_none_values(llm_result)
            llm_result = self.minified(**llm_result)

        dicto = llm_result.model_dump(by_alias=True)
        return self.original_type(**dicto)

    def _get_short_field_name(self, field_name: str) -> str:
        new_name = self.small_names_list.pop(0)
        self.field_names_mapper[field_name] = new_name
        return new_name

    def _get_small_name_list(self, max_size: int = 3) -> list[str]:
        result = []
        root_chars = ["a", "b", "c"]
        chars = list("abcdefghijklmnopqrstuvwxyz")
        result.extend(chars)

        for root_size in (1, max_size - 1):
            for char in root_chars:
                for sub_char in chars:
                    result.append(char * root_size + sub_char)

        return result
