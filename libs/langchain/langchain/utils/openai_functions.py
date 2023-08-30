from typing import Optional, Type, TypedDict

from langchain.pydantic_v1 import BaseModel
from langchain.utils.json_schema import dereference_refs


class FunctionDescription(TypedDict):
    """Representation of a callable function to the OpenAI API."""

    name: str
    """The name of the function."""
    description: str
    """A description of the function."""
    parameters: dict
    """The parameters of the function."""


def convert_pydantic_to_openai_function(
    model: Type[BaseModel],
    *,
    name: Optional[str] = None,
    description: Optional[str] = None
) -> FunctionDescription:
    schema = dereference_refs(model.schema())
    schema.pop("definitions", None)
    return {
        "name": name or schema["title"],
        "description": description or schema["description"],
        "parameters": schema,
    }
