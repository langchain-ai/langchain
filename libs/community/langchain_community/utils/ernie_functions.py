from typing import Literal, Optional, Type, TypedDict

from langchain_core.pydantic_v1 import BaseModel
from langchain_core.utils.json_schema import dereference_refs


class FunctionDescription(TypedDict):
    """Representation of a callable function to the Ernie API."""

    name: str
    """The name of the function."""
    description: str
    """A description of the function."""
    parameters: dict
    """The parameters of the function."""


class ToolDescription(TypedDict):
    """Representation of a callable function to the Ernie API."""

    type: Literal["function"]
    function: FunctionDescription


def convert_pydantic_to_ernie_function(
    model: Type[BaseModel],
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> FunctionDescription:
    """Convert a Pydantic model to a function description for the Ernie API."""
    schema = dereference_refs(model.schema())
    schema.pop("definitions", None)
    return {
        "name": name or schema["title"],
        "description": description or schema["description"],
        "parameters": schema,
    }


def convert_pydantic_to_ernie_tool(
    model: Type[BaseModel],
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> ToolDescription:
    """Convert a Pydantic model to a function description for the Ernie API."""
    function = convert_pydantic_to_ernie_function(
        model, name=name, description=description
    )
    return {"type": "function", "function": function}
