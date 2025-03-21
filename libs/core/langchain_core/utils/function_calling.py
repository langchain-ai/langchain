"""Methods for creating function specs in the style of OpenAI Functions."""

from __future__ import annotations

import collections
import inspect
import logging
import types
import typing
import uuid
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    Literal,
    Optional,
    Union,
    cast,
)

from pydantic import BaseModel
from pydantic.v1 import BaseModel as BaseModelV1
from typing_extensions import TypedDict, get_args, get_origin, is_typeddict

from langchain_core._api import beta, deprecated
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.utils.json_schema import dereference_refs
from langchain_core.utils.pydantic import is_basemodel_subclass

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
    """The type of the tool."""
    function: FunctionDescription
    """The function description."""


def _rm_titles(kv: dict, prev_key: str = "") -> dict:
    new_kv = {}
    for k, v in kv.items():
        if k == "title":
            if isinstance(v, dict) and prev_key == "properties" and "title" in v:
                new_kv[k] = _rm_titles(v, k)
            else:
                continue
        elif isinstance(v, dict):
            new_kv[k] = _rm_titles(v, k)
        else:
            new_kv[k] = v
    return new_kv


def _convert_json_schema_to_openai_function(
    schema: dict,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    rm_titles: bool = True,
) -> FunctionDescription:
    """Converts a Pydantic model to a function description for the OpenAI API.

    Args:
        schema: The JSON schema to convert.
        name: The name of the function. If not provided, the title of the schema will be
            used.
        description: The description of the function. If not provided, the description
            of the schema will be used.
        rm_titles: Whether to remove titles from the schema. Defaults to True.

    Returns:
        The function description.
    """
    schema = dereference_refs(schema)
    if "definitions" in schema:  # pydantic 1
        schema.pop("definitions", None)
    if "$defs" in schema:  # pydantic 2
        schema.pop("$defs", None)
    title = schema.pop("title", "")
    default_description = schema.pop("description", "")
    return {
        "name": name or title,
        "description": description or default_description,
        "parameters": _rm_titles(schema) if rm_titles else schema,
    }


def _convert_pydantic_to_openai_function(
    model: type,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    rm_titles: bool = True,
) -> FunctionDescription:
    """Converts a Pydantic model to a function description for the OpenAI API.

    Args:
        model: The Pydantic model to convert.
        name: The name of the function. If not provided, the title of the schema will be
            used.
        description: The description of the function. If not provided, the description
            of the schema will be used.
        rm_titles: Whether to remove titles from the schema. Defaults to True.

    Returns:
        The function description.
    """
    if hasattr(model, "model_json_schema"):
        schema = model.model_json_schema()  # Pydantic 2
    elif hasattr(model, "schema"):
        schema = model.schema()  # Pydantic 1
    else:
        msg = "Model must be a Pydantic model."
        raise TypeError(msg)
    return _convert_json_schema_to_openai_function(
        schema, name=name, description=description, rm_titles=rm_titles
    )


convert_pydantic_to_openai_function = deprecated(
    "0.1.16",
    alternative="langchain_core.utils.function_calling.convert_to_openai_function()",
    removal="1.0",
)(_convert_pydantic_to_openai_function)


@deprecated(
    "0.1.16",
    alternative="langchain_core.utils.function_calling.convert_to_openai_tool()",
    removal="1.0",
)
def convert_pydantic_to_openai_tool(
    model: type[BaseModel],
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> ToolDescription:
    """Converts a Pydantic model to a function description for the OpenAI API.

    Args:
        model: The Pydantic model to convert.
        name: The name of the function. If not provided, the title of the schema will be
            used.
        description: The description of the function. If not provided, the description
            of the schema will be used.

    Returns:
        The tool description.
    """
    function = _convert_pydantic_to_openai_function(
        model, name=name, description=description
    )
    return {"type": "function", "function": function}


def _get_python_function_name(function: Callable) -> str:
    """Get the name of a Python function."""
    return function.__name__


def _convert_python_function_to_openai_function(
    function: Callable,
) -> FunctionDescription:
    """Convert a Python function to an OpenAI function-calling API compatible dict.

    Assumes the Python function has type hints and a docstring with a description. If
        the docstring has Google Python style argument descriptions, these will be
        included as well.

    Args:
        function: The Python function to convert.

    Returns:
        The OpenAI function description.
    """
    from langchain_core.tools.base import create_schema_from_function

    func_name = _get_python_function_name(function)
    model = create_schema_from_function(
        func_name,
        function,
        filter_args=(),
        parse_docstring=True,
        error_on_invalid_docstring=False,
        include_injected=False,
    )
    return _convert_pydantic_to_openai_function(
        model,
        name=func_name,
        description=model.__doc__,
    )


convert_python_function_to_openai_function = deprecated(
    "0.1.16",
    alternative="langchain_core.utils.function_calling.convert_to_openai_function()",
    removal="1.0",
)(_convert_python_function_to_openai_function)


def _convert_typed_dict_to_openai_function(typed_dict: type) -> FunctionDescription:
    visited: dict = {}
    from pydantic.v1 import BaseModel

    model = cast(
        type[BaseModel],
        _convert_any_typed_dicts_to_pydantic(typed_dict, visited=visited),
    )
    return _convert_pydantic_to_openai_function(model)  # type: ignore


_MAX_TYPED_DICT_RECURSION = 25


def _convert_any_typed_dicts_to_pydantic(
    type_: type,
    *,
    visited: dict,
    depth: int = 0,
) -> type:
    from pydantic.v1 import Field as Field_v1
    from pydantic.v1 import create_model as create_model_v1

    if type_ in visited:
        return visited[type_]
    elif depth >= _MAX_TYPED_DICT_RECURSION:
        return type_
    elif is_typeddict(type_):
        typed_dict = type_
        docstring = inspect.getdoc(typed_dict)
        annotations_ = typed_dict.__annotations__
        description, arg_descriptions = _parse_google_docstring(
            docstring, list(annotations_)
        )
        fields: dict = {}
        for arg, arg_type in annotations_.items():
            if get_origin(arg_type) is Annotated:
                annotated_args = get_args(arg_type)
                new_arg_type = _convert_any_typed_dicts_to_pydantic(
                    annotated_args[0], depth=depth + 1, visited=visited
                )
                field_kwargs = dict(zip(("default", "description"), annotated_args[1:]))
                if (field_desc := field_kwargs.get("description")) and not isinstance(
                    field_desc, str
                ):
                    msg = (
                        f"Invalid annotation for field {arg}. Third argument to "
                        f"Annotated must be a string description, received value of "
                        f"type {type(field_desc)}."
                    )
                    raise ValueError(msg)
                elif arg_desc := arg_descriptions.get(arg):
                    field_kwargs["description"] = arg_desc
                else:
                    pass
                fields[arg] = (new_arg_type, Field_v1(**field_kwargs))
            else:
                new_arg_type = _convert_any_typed_dicts_to_pydantic(
                    arg_type, depth=depth + 1, visited=visited
                )
                field_kwargs = {"default": ...}
                if arg_desc := arg_descriptions.get(arg):
                    field_kwargs["description"] = arg_desc
                fields[arg] = (new_arg_type, Field_v1(**field_kwargs))
        model = create_model_v1(typed_dict.__name__, **fields)
        model.__doc__ = description
        visited[typed_dict] = model
        return model
    elif (origin := get_origin(type_)) and (type_args := get_args(type_)):
        subscriptable_origin = _py_38_safe_origin(origin)
        type_args = tuple(
            _convert_any_typed_dicts_to_pydantic(arg, depth=depth + 1, visited=visited)
            for arg in type_args  # type: ignore[index]
        )
        return subscriptable_origin[type_args]  # type: ignore[index]
    else:
        return type_


def _format_tool_to_openai_function(tool: BaseTool) -> FunctionDescription:
    """Format tool into the OpenAI function API.

    Args:
        tool: The tool to format.

    Returns:
        The function description.
    """
    from langchain_core.tools import simple

    is_simple_oai_tool = isinstance(tool, simple.Tool) and not tool.args_schema
    if tool.tool_call_schema and not is_simple_oai_tool:
        if isinstance(tool.tool_call_schema, dict):
            return _convert_json_schema_to_openai_function(
                tool.tool_call_schema, name=tool.name, description=tool.description
            )
        elif issubclass(tool.tool_call_schema, (BaseModel, BaseModelV1)):
            return _convert_pydantic_to_openai_function(
                tool.tool_call_schema, name=tool.name, description=tool.description
            )
        else:
            error_msg = (
                f"Unsupported tool call schema: {tool.tool_call_schema}. "
                "Tool call schema must be a JSON schema dict or a Pydantic model."
            )
            raise ValueError(error_msg)
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


format_tool_to_openai_function = deprecated(
    "0.1.16",
    alternative="langchain_core.utils.function_calling.convert_to_openai_function()",
    removal="1.0",
)(_format_tool_to_openai_function)


@deprecated(
    "0.1.16",
    alternative="langchain_core.utils.function_calling.convert_to_openai_tool()",
    removal="1.0",
)
def format_tool_to_openai_tool(tool: BaseTool) -> ToolDescription:
    """Format tool into the OpenAI function API.

    Args:
        tool: The tool to format.

    Returns:
        The tool description.
    """
    function = _format_tool_to_openai_function(tool)
    return {"type": "function", "function": function}


def convert_to_openai_function(
    function: Union[dict[str, Any], type, Callable, BaseTool],
    *,
    strict: Optional[bool] = None,
) -> dict[str, Any]:
    """Convert a raw function/class to an OpenAI function.

    Args:
        function:
            A dictionary, Pydantic BaseModel class, TypedDict class, a LangChain
            Tool object, or a Python function. If a dictionary is passed in, it is
            assumed to already be a valid OpenAI function, a JSON schema with
            top-level 'title' key specified, an Anthropic format
            tool, or an Amazon Bedrock Converse format tool.
        strict:
            If True, model output is guaranteed to exactly match the JSON Schema
            provided in the function definition. If None, ``strict`` argument will not
            be included in function definition.

    Returns:
        A dict version of the passed in function which is compatible with the OpenAI
        function-calling API.

    Raises:
        ValueError: If function is not in a supported format.

    .. versionchanged:: 0.2.29

        ``strict`` arg added.

    .. versionchanged:: 0.3.13

        Support for Anthropic format tools added.

    .. versionchanged:: 0.3.14

        Support for Amazon Bedrock Converse format tools added.

    .. versionchanged:: 0.3.16

        'description' and 'parameters' keys are now optional. Only 'name' is
        required and guaranteed to be part of the output.
    """
    from langchain_core.tools import BaseTool

    # an Anthropic format tool
    if isinstance(function, dict) and all(
        k in function for k in ("name", "input_schema")
    ):
        oai_function = {
            "name": function["name"],
            "parameters": function["input_schema"],
        }
        if "description" in function:
            oai_function["description"] = function["description"]
    # an Amazon Bedrock Converse format tool
    elif isinstance(function, dict) and "toolSpec" in function:
        oai_function = {
            "name": function["toolSpec"]["name"],
            "parameters": function["toolSpec"]["inputSchema"]["json"],
        }
        if "description" in function["toolSpec"]:
            oai_function["description"] = function["toolSpec"]["description"]
    # already in OpenAI function format
    elif isinstance(function, dict) and "name" in function:
        oai_function = {
            k: v
            for k, v in function.items()
            if k in ("name", "description", "parameters", "strict")
        }
    # a JSON schema with title and description
    elif isinstance(function, dict) and "title" in function:
        function_copy = function.copy()
        oai_function = {"name": function_copy.pop("title")}
        if "description" in function_copy:
            oai_function["description"] = function_copy.pop("description")
        if function_copy and "properties" in function_copy:
            oai_function["parameters"] = function_copy
    elif isinstance(function, type) and is_basemodel_subclass(function):
        oai_function = cast(dict, _convert_pydantic_to_openai_function(function))
    elif is_typeddict(function):
        oai_function = cast(
            dict, _convert_typed_dict_to_openai_function(cast(type, function))
        )
    elif isinstance(function, BaseTool):
        oai_function = cast(dict, _format_tool_to_openai_function(function))
    elif callable(function):
        oai_function = cast(dict, _convert_python_function_to_openai_function(function))
    else:
        msg = (
            f"Unsupported function\n\n{function}\n\nFunctions must be passed in"
            " as Dict, pydantic.BaseModel, or Callable. If they're a dict they must"
            " either be in OpenAI function format or valid JSON schema with top-level"
            " 'title' and 'description' keys."
        )
        raise ValueError(msg)

    if strict is not None:
        if "strict" in oai_function and oai_function["strict"] != strict:
            msg = (
                f"Tool/function already has a 'strict' key wth value "
                f"{oai_function['strict']} which is different from the explicit "
                f"`strict` arg received {strict=}."
            )
            raise ValueError(msg)
        oai_function["strict"] = strict
        if strict:
            # As of 08/06/24, OpenAI requires that additionalProperties be supplied and
            # set to False if strict is True.
            # All properties layer needs 'additionalProperties=False'
            oai_function["parameters"] = _recursive_set_additional_properties_false(
                oai_function["parameters"]
            )
    return oai_function


def convert_to_openai_tool(
    tool: Union[dict[str, Any], type[BaseModel], Callable, BaseTool],
    *,
    strict: Optional[bool] = None,
) -> dict[str, Any]:
    """Convert a tool-like object to an OpenAI tool schema.

    OpenAI tool schema reference:
    https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools

    Args:
        tool:
            Either a dictionary, a pydantic.BaseModel class, Python function, or
            BaseTool. If a dictionary is passed in, it is
            assumed to already be a valid OpenAI function, a JSON schema with
            top-level 'title' key specified, an Anthropic format
            tool, or an Amazon Bedrock Converse format tool.
        strict:
            If True, model output is guaranteed to exactly match the JSON Schema
            provided in the function definition. If None, ``strict`` argument will not
            be included in tool definition.

    Returns:
        A dict version of the passed in tool which is compatible with the
        OpenAI tool-calling API.

    .. versionchanged:: 0.2.29

        ``strict`` arg added.

    .. versionchanged:: 0.3.13

        Support for Anthropic format tools added.

    .. versionchanged:: 0.3.14

        Support for Amazon Bedrock Converse format tools added.

    .. versionchanged:: 0.3.16

        'description' and 'parameters' keys are now optional. Only 'name' is
        required and guaranteed to be part of the output.
    """
    if isinstance(tool, dict) and tool.get("type") == "function" and "function" in tool:
        return tool
    oai_function = convert_to_openai_function(tool, strict=strict)
    return {"type": "function", "function": oai_function}


def convert_to_json_schema(
    schema: Union[dict[str, Any], type[BaseModel], Callable, BaseTool],
    *,
    strict: Optional[bool] = None,
) -> dict[str, Any]:
    """Convert a schema representation to a JSON schema."""
    openai_tool = convert_to_openai_tool(schema, strict=strict)
    if (
        not isinstance(openai_tool, dict)
        or "function" not in openai_tool
        or "name" not in openai_tool["function"]
    ):
        error_message = "Input must be a valid OpenAI-format tool."
        raise ValueError(error_message)

    openai_function = openai_tool["function"]
    json_schema = {}
    json_schema["title"] = openai_function["name"]

    if "description" in openai_function:
        json_schema["description"] = openai_function["description"]

    if "parameters" in openai_function:
        parameters = openai_function["parameters"].copy()
        json_schema.update(parameters)

    return json_schema


@beta()
def tool_example_to_messages(
    input: str,
    tool_calls: list[BaseModel],
    tool_outputs: Optional[list[str]] = None,
    *,
    ai_response: Optional[str] = None,
) -> list[BaseMessage]:
    """Convert an example into a list of messages that can be fed into an LLM.

    This code is an adapter that converts a single example to a list of messages
    that can be fed into a chat model.

    The list of messages per example by default corresponds to:

    1) HumanMessage: contains the content from which content should be extracted.
    2) AIMessage: contains the extracted information from the model
    3) ToolMessage: contains confirmation to the model that the model requested a tool
        correctly.

    If `ai_response` is specified, there will be a final AIMessage with that response.

    The ToolMessage is required because some chat models are hyper-optimized for agents
    rather than for an extraction use case.

    Arguments:
        input: string, the user input
        tool_calls: List[BaseModel], a list of tool calls represented as Pydantic
            BaseModels
        tool_outputs: Optional[List[str]], a list of tool call outputs.
            Does not need to be provided. If not provided, a placeholder value
            will be inserted. Defaults to None.
        ai_response: Optional[str], if provided, content for a final AIMessage.

    Returns:
        A list of messages

    Examples:

        .. code-block:: python

            from typing import List, Optional
            from pydantic import BaseModel, Field
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
    messages: list[BaseMessage] = [HumanMessage(content=input)]
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
                    "arguments": tool_call.model_dump_json(),
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

    if ai_response:
        messages.append(AIMessage(content=ai_response))
    return messages


def _parse_google_docstring(
    docstring: Optional[str],
    args: list[str],
    *,
    error_on_invalid_docstring: bool = False,
) -> tuple[str, dict]:
    """Parse the function and argument descriptions from the docstring of a function.

    Assumes the function docstring follows Google Python style guide.
    """
    if docstring:
        docstring_blocks = docstring.split("\n\n")
        if error_on_invalid_docstring:
            filtered_annotations = {
                arg for arg in args if arg not in ("run_manager", "callbacks", "return")
            }
            if filtered_annotations and (
                len(docstring_blocks) < 2
                or not any(block.startswith("Args:") for block in docstring_blocks[1:])
            ):
                msg = "Found invalid Google-Style docstring."
                raise ValueError(msg)
        descriptors = []
        args_block = None
        past_descriptors = False
        for block in docstring_blocks:
            if block.startswith("Args:"):
                args_block = block
                break
            elif block.startswith(("Returns:", "Example:")):
                # Don't break in case Args come after
                past_descriptors = True
            elif not past_descriptors:
                descriptors.append(block)
            else:
                continue
        description = " ".join(descriptors)
    else:
        if error_on_invalid_docstring:
            msg = "Found invalid Google-Style docstring."
            raise ValueError(msg)
        description = ""
        args_block = None
    arg_descriptions = {}
    if args_block:
        arg = None
        for line in args_block.split("\n")[1:]:
            if ":" in line:
                arg, desc = line.split(":", maxsplit=1)
                arg = arg.strip()
                arg_name, _, _annotations = arg.partition(" ")
                if _annotations.startswith("(") and _annotations.endswith(")"):
                    arg = arg_name
                arg_descriptions[arg] = desc.strip()
            elif arg:
                arg_descriptions[arg] += " " + line.strip()
    return description, arg_descriptions


def _py_38_safe_origin(origin: type) -> type:
    origin_union_type_map: dict[type, Any] = (
        {types.UnionType: Union} if hasattr(types, "UnionType") else {}
    )

    origin_map: dict[type, Any] = {
        dict: dict,
        list: list,
        tuple: tuple,
        set: set,
        collections.abc.Iterable: typing.Iterable,
        collections.abc.Mapping: typing.Mapping,
        collections.abc.Sequence: typing.Sequence,
        collections.abc.MutableMapping: typing.MutableMapping,
        **origin_union_type_map,
    }
    return cast(type, origin_map.get(origin, origin))


def _recursive_set_additional_properties_false(
    schema: dict[str, Any],
) -> dict[str, Any]:
    if isinstance(schema, dict):
        # Check if 'required' is a key at the current level or if the schema is empty,
        # in which case additionalProperties still needs to be specified.
        if "required" in schema or (
            "properties" in schema and not schema["properties"]
        ):
            schema["additionalProperties"] = False

        # Recursively check 'properties' and 'items' if they exist
        if "properties" in schema:
            for value in schema["properties"].values():
                _recursive_set_additional_properties_false(value)
        if "items" in schema:
            _recursive_set_additional_properties_false(schema["items"])

    return schema
