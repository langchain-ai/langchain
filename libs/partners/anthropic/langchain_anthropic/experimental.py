import json
from typing import (
    Any,
    Dict,
    List,
    Union,
)

from langchain_core._api import deprecated
from langchain_core.pydantic_v1 import Field

from langchain_anthropic.chat_models import ChatAnthropic

SYSTEM_PROMPT_FORMAT = """In this environment you have access to a set of tools you can use to answer the user's question.

You may call them like this:
<function_calls>
<invoke>
<tool_name>$TOOL_NAME</tool_name>
<parameters>
<$PARAMETER_NAME>$PARAMETER_VALUE</$PARAMETER_NAME>
...
</parameters>
</invoke>
</function_calls>

Here are the tools available:
<tools>
{formatted_tools}
</tools>"""  # noqa: E501

TOOL_FORMAT = """<tool_description>
<tool_name>{tool_name}</tool_name>
<description>{tool_description}</description>
<parameters>
{formatted_parameters}
</parameters>
</tool_description>"""

TOOL_PARAMETER_FORMAT = """<parameter>
<name>{parameter_name}</name>
<type>{parameter_type}</type>
<description>{parameter_description}</description>
</parameter>"""


def _get_type(parameter: Dict[str, Any]) -> str:
    if "type" in parameter:
        return parameter["type"]
    if "anyOf" in parameter:
        return json.dumps({"anyOf": parameter["anyOf"]})
    if "allOf" in parameter:
        return json.dumps({"allOf": parameter["allOf"]})
    return json.dumps(parameter)


def get_system_message(tools: List[Dict]) -> str:
    """Generate a system message that describes the available tools."""
    tools_data: List[Dict] = [
        {
            "tool_name": tool["name"],
            "tool_description": tool["description"],
            "formatted_parameters": "\n".join(
                [
                    TOOL_PARAMETER_FORMAT.format(
                        parameter_name=name,
                        parameter_type=_get_type(parameter),
                        parameter_description=parameter.get("description"),
                    )
                    for name, parameter in tool["parameters"]["properties"].items()
                ]
            ),
        }
        for tool in tools
    ]
    tools_formatted = "\n".join(
        [
            TOOL_FORMAT.format(
                tool_name=tool["tool_name"],
                tool_description=tool["tool_description"],
                formatted_parameters=tool["formatted_parameters"],
            )
            for tool in tools_data
        ]
    )
    return SYSTEM_PROMPT_FORMAT.format(formatted_tools=tools_formatted)


def _xml_to_dict(t: Any) -> Union[str, Dict[str, Any]]:
    # Base case: If the element has no children, return its text or an empty string.
    if len(t) == 0:
        return t.text or ""

    # Recursive case: The element has children. Convert them into a dictionary.
    d: Dict[str, Any] = {}
    for child in t:
        if child.tag not in d:
            d[child.tag] = _xml_to_dict(child)
        else:
            # Handle multiple children with the same tag
            if not isinstance(d[child.tag], list):
                d[child.tag] = [d[child.tag]]  # Convert existing entry into a list
            d[child.tag].append(_xml_to_dict(child))
    return d


def _xml_to_function_call(invoke: Any, tools: List[Dict]) -> Dict[str, Any]:
    name = invoke.find("tool_name").text
    arguments = _xml_to_dict(invoke.find("parameters"))

    # make list elements in arguments actually lists
    filtered_tools = [tool for tool in tools if tool["name"] == name]
    if len(filtered_tools) > 0 and not isinstance(arguments, str):
        tool = filtered_tools[0]
        for key, value in arguments.items():
            if key in tool["parameters"]["properties"]:
                if "type" in tool["parameters"]["properties"][key]:
                    if tool["parameters"]["properties"][key][
                        "type"
                    ] == "array" and not isinstance(value, list):
                        arguments[key] = [value]
                    if (
                        tool["parameters"]["properties"][key]["type"] != "object"
                        and isinstance(value, dict)
                        and len(value.keys()) == 1
                    ):
                        arguments[key] = list(value.values())[0]

    return {
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
        "type": "function",
    }


def _xml_to_tool_calls(elem: Any, tools: List[Dict]) -> List[Dict[str, Any]]:
    """
    Convert an XML element and its children into a dictionary of dictionaries.
    """
    invokes = elem.findall("invoke")

    return [_xml_to_function_call(invoke, tools) for invoke in invokes]


@deprecated(
    "0.1.5",
    removal="0.3.0",
    alternative="ChatAnthropic",
    message=(
        "Tool-calling is now officially supported by the Anthropic API so this "
        "workaround is no longer needed."
    ),
)
class ChatAnthropicTools(ChatAnthropic):
    """Chat model for interacting with Anthropic functions."""

    _xmllib: Any = Field(default=None)
