"""Methods for creating function specs in the style of OpenAI Functions"""

from __future__ import annotations

import logging
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Type,
    Union,
    cast,
)

from typing_extensions import TypedDict

from langchain_core._api import deprecated
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.utils.json_schema import dereference_refs

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
logger = logging.getLogger(__name__)
PYTHON_TO_JSON_TYPES = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
}


class FunctionDescription(TypedDict):
    """Representation of a callable function to send to an LLM."""

    name: str
    """The name of the function."""
    description: str
    """A description of the function."""
    parameters: dict
    """The parameters of the function."""


class ToolDescription(TypedDict):
    """Representation of a callable function to the OpenAI API."""

    type: Literal["function"]
    function: FunctionDescription


def _rm_titles(kv: dict, prev_key: str = "") -> dict:
    new_kv = {}
    for k, v in kv.items():
        if k == "title":
            if isinstance(v, dict) and prev_key == "properties" and "title" in v.keys():
                new_kv[k] = _rm_titles(v, k)
            else:
                continue
        elif isinstance(v, dict):
            new_kv[k] = _rm_titles(v, k)
        else:
            new_kv[k] = v
    return new_kv


@deprecated(
    "0.1.16",
    alternative="langchain_core.utils.function_calling.convert_to_openai_function()",
    removal="0.3.0",
)
def convert_pydantic_to_openai_function(
    model: Type[BaseModel],
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    rm_titles: bool = True,
) -> FunctionDescription:
    """Converts a Pydantic model to a function description for the OpenAI API."""
    schema = dereference_refs(model.schema())
    schema.pop("definitions", None)
    title = schema.pop("title", "")
    default_description = schema.pop("description", "")
    return {
        "name": name or title,
        "description": description or default_description,
        "parameters": _rm_titles(schema) if rm_titles else schema,
    }


@deprecated(
    "0.1.16",
    alternative="langchain_core.utils.function_calling.convert_to_openai_tool()",
    removal="0.3.0",
)
def convert_pydantic_to_openai_tool(
    model: Type[BaseModel],
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> ToolDescription:
    """Converts a Pydantic model to a function description for the OpenAI API."""
    function = convert_pydantic_to_openai_function(
        model, name=name, description=description
    )
    return {"type": "function", "function": function}


def _get_python_function_name(function: Callable) -> str:
    """Get the name of a Python function."""
    return function.__name__


@deprecated(
    "0.1.16",
    alternative="langchain_core.utils.function_calling.convert_to_openai_function()",
    removal="0.3.0",
)
def convert_python_function_to_openai_function(
    function: Callable,
) -> FunctionDescription:
    """Convert a Python function to an OpenAI function-calling API compatible dict.

    Assumes the Python function has type hints and a docstring with a description. If
        the docstring has Google Python style argument descriptions, these will be
        included as well.
    """
    from langchain_core import tools

    func_name = _get_python_function_name(function)
    model = tools.create_schema_from_function(
        func_name, function, filter_args=(), parse_docstring=True
    )
    return convert_pydantic_to_openai_function(
        model,
        name=func_name,
        description=model.__doc__,
    )


@deprecated(
    "0.1.16",
    alternative="langchain_core.utils.function_calling.convert_to_openai_function()",
    removal="0.3.0",
)
def format_tool_to_openai_function(tool: BaseTool) -> FunctionDescription:
    """Format tool into the OpenAI function API."""
    if tool.args_schema:
        return convert_pydantic_to_openai_function(
            tool.args_schema, name=tool.name, description=tool.description
        )
    else:
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                # This is a hack to get around the fact that some tools
                # do not expose an args_schema, and expect an argument
                # which is a string.
                # And Open AI does not support an array type for the
                # parameters.
                "properties": {
                    "__arg1": {"title": "__arg1", "type": "string"},
                },
                "required": ["__arg1"],
                "type": "object",
            },
        }


@deprecated(
    "0.1.16",
    alternative="langchain_core.utils.function_calling.convert_to_openai_tool()",
    removal="0.3.0",
)
def format_tool_to_openai_tool(tool: BaseTool) -> ToolDescription:
    """Format tool into the OpenAI function API."""
    function = format_tool_to_openai_function(tool)
    return {"type": "function", "function": function}


def convert_to_openai_function(
    function: Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool],
) -> Dict[str, Any]:
    """Convert a raw function/class to an OpenAI function.

    Args:
        function: Either a dictionary, a pydantic.BaseModel class, or a Python function.
            If a dictionary is passed in, it is assumed to already be a valid OpenAI
            function or a JSON schema with top-level 'title' and 'description' keys
            specified.

    Returns:
        A dict version of the passed in function which is compatible with the
            OpenAI function-calling API.
    """
    from langchain_core.tools import BaseTool

    # already in OpenAI function format
    if isinstance(function, dict) and all(
        k in function for k in ("name", "description", "parameters")
    ):
        return function
    # a JSON schema with title and description
    elif isinstance(function, dict) and all(
        k in function for k in ("title", "description", "properties")
    ):
        function = function.copy()
        return {
            "name": function.pop("title"),
            "description": function.pop("description"),
            "parameters": function,
        }
    elif isinstance(function, type) and issubclass(function, BaseModel):
        return cast(Dict, convert_pydantic_to_openai_function(function))
    elif isinstance(function, BaseTool):
        return cast(Dict, format_tool_to_openai_function(function))
    elif callable(function):
        return cast(Dict, convert_python_function_to_openai_function(function))
    else:
        raise ValueError(
            f"Unsupported function\n\n{function}\n\nFunctions must be passed in"
            " as Dict, pydantic.BaseModel, or Callable. If they're a dict they must"
            " either be in OpenAI function format or valid JSON schema with top-level"
            " 'title' and 'description' keys."
        )


def convert_to_openai_tool(
    tool: Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool],
) -> Dict[str, Any]:
    """Convert a raw function/class to an OpenAI tool.

    Args:
        tool: Either a dictionary, a pydantic.BaseModel class, Python function, or
            BaseTool. If a dictionary is passed in, it is assumed to already be a valid
            OpenAI tool, OpenAI function, or a JSON schema with top-level 'title' and
            'description' keys specified.

    Returns:
        A dict version of the passed in tool which is compatible with the
            OpenAI tool-calling API.
    """
    if isinstance(tool, dict) and tool.get("type") == "function" and "function" in tool:
        return tool
    function = convert_to_openai_function(tool)
    return {"type": "function", "function": function}


def tool_example_to_messages(
    input: str, tool_calls: List[BaseModel], tool_outputs: Optional[List[str]] = None
) -> List[BaseMessage]:
    """Convert an example into a list of messages that can be fed into an LLM.

    This code is an adapter that converts a single example to a list of messages
    that can be fed into a chat model.

    The list of messages per example corresponds to:

    1) HumanMessage: contains the content from which content should be extracted.
    2) AIMessage: contains the extracted information from the model
    3) ToolMessage: contains confirmation to the model that the model requested a tool
        correctly.

    The ToolMessage is required because some chat models are hyper-optimized for agents
    rather than for an extraction use case.

    Arguments:
        input: string, the user input
        tool_calls: List[BaseModel], a list of tool calls represented as Pydantic
            BaseModels
        tool_outputs: Optional[List[str]], a list of tool call outputs.
            Does not need to be provided. If not provided, a placeholder value
            will be inserted.

    Returns:
        A list of messages

    Examples:

        .. code-block:: python

            from typing import List, Optional
            from langchain_core.pydantic_v1 import BaseModel, Field
            from langchain_openai import ChatOpenAI

            class Person(BaseModel):
                '''Information about a person.'''
                name: Optional[str] = Field(..., description="The name of the person")
                hair_color: Optional[str] = Field(
                    ..., description="The color of the person's hair if known"
                )
                height_in_meters: Optional[str] = Field(
                    ..., description="Height in METERs"
                )

            examples = [
                (
                    "The ocean is vast and blue. It's more than 20,000 feet deep.",
                    Person(name=None, height_in_meters=None, hair_color=None),
                ),
                (
                    "Fiona traveled far from France to Spain.",
                    Person(name="Fiona", height_in_meters=None, hair_color=None),
                ),
            ]


            messages = []

            for txt, tool_call in examples:
                messages.extend(
                    tool_example_to_messages(txt, [tool_call])
                )
    """
    messages: List[BaseMessage] = [HumanMessage(content=input)]
    openai_tool_calls = []
    for tool_call in tool_calls:
        openai_tool_calls.append(
            {
                "id": str(uuid.uuid4()),
                "type": "function",
                "function": {
                    # The name of the function right now corresponds to the name
                    # of the pydantic model. This is implicit in the API right now,
                    # and will be improved over time.
                    "name": tool_call.__class__.__name__,
                    "arguments": tool_call.json(),
                },
            }
        )
    messages.append(
        AIMessage(content="", additional_kwargs={"tool_calls": openai_tool_calls})
    )
    tool_outputs = tool_outputs or ["You have correctly called this tool."] * len(
        openai_tool_calls
    )
    for output, tool_call_dict in zip(tool_outputs, openai_tool_calls):
        messages.append(ToolMessage(content=output, tool_call_id=tool_call_dict["id"]))  # type: ignore
    return messages
