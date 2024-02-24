import copy
import json
from json import JSONDecodeError
from typing import Any, List, Type

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseGenerationOutputParser
from langchain_core.output_parsers.json import parse_partial_json
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.pydantic_v1 import BaseModel


class JsonOutputToolsParser(BaseGenerationOutputParser[Any]):
    """Parse tools from OpenAI response."""

    strict: bool = False
    """Whether to allow non-JSON-compliant strings.

    See: https://docs.python.org/3/library/json.html#encoders-and-decoders

    Useful when the parsed output may include unicode characters or new lines.
    """
    return_id: bool = False
    """Whether to return the tool call id."""
    first_tool_only: bool = False
    """Whether to return only the first tool call.
    
    If False, the result will be a list of tool calls, or an empty list 
    if no tool calls are found.
    
    If true, and multiple tool calls are found, only the first one will be returned,
    and the other tool calls will be ignored. 
    If no tool calls are found, None will be returned. 
    """

    def parse_result(self, result: List[Generation], *, partial: bool = False) -> Any:
        generation = result[0]
        if not isinstance(generation, ChatGeneration):
            raise OutputParserException(
                "This output parser can only be used with a chat generation."
            )
        message = generation.message
        try:
            tool_calls = copy.deepcopy(message.additional_kwargs["tool_calls"])
        except KeyError:
            return []

        final_tools = []
        exceptions = []
        for tool_call in tool_calls:
            if "function" not in tool_call:
                continue
            try:
                if partial:
                    function_args = parse_partial_json(
                        tool_call["function"]["arguments"], strict=self.strict
                    )
                else:
                    function_args = json.loads(
                        tool_call["function"]["arguments"], strict=self.strict
                    )
            except JSONDecodeError as e:
                exceptions.append(
                    f"Function {tool_call['function']['name']} arguments:\n\n"
                    f"{tool_call['function']['arguments']}\n\nare not valid JSON. "
                    f"Received JSONDecodeError {e}"
                )
                continue
            parsed = {
                "type": tool_call["function"]["name"],
                "args": function_args,
            }
            if self.return_id:
                parsed["id"] = tool_call["id"]
            final_tools.append(parsed)
        if exceptions:
            raise OutputParserException("\n\n".join(exceptions))
        if self.first_tool_only:
            return final_tools[0] if final_tools else None
        return final_tools


class JsonOutputKeyToolsParser(JsonOutputToolsParser):
    """Parse tools from OpenAI response."""

    key_name: str
    """The type of tools to return."""

    def __init__(self, key_name: str, **kwargs: Any) -> None:
        """Allow init with positional args."""
        # Backwards compatibility for old argument name.
        if "return_single" in kwargs:
            if not kwargs.get("first_tool_only"):
                kwargs["first_tool_only"] = kwargs.pop("return_single")
            else:
                raise ValueError(
                    "Cannot use both 'return_single' and 'first_tool_only' arguments."
                )
        super().__init__(key_name=key_name, **kwargs)

    def parse_result(self, result: List[Generation], *, partial: bool = False) -> Any:
        parsed_result = super().parse_result(result, partial=partial)
        if self.first_tool_only:
            single_result = (
                parsed_result
                if parsed_result and parsed_result["type"] == self.key_name
                else None
            )
            if self.return_id:
                return single_result
            elif single_result:
                return single_result["args"]
            else:
                return None
        parsed_result = [res for res in parsed_result if res["type"] == self.key_name]
        if not self.return_id:
            parsed_result = [res["args"] for res in parsed_result]
        return parsed_result


class PydanticToolsParser(JsonOutputToolsParser):
    """Parse tools from OpenAI response."""

    tools: List[Type[BaseModel]]

    def parse_result(self, result: List[Generation], *, partial: bool = False) -> Any:
        parsed_result = super().parse_result(result, partial=partial)
        name_dict = {tool.__name__: tool for tool in self.tools}
        if self.first_tool_only:
            return (
                name_dict[parsed_result["type"]](**parsed_result["args"])
                if parsed_result
                else None
            )
        return [name_dict[res["type"]](**res["args"]) for res in parsed_result]
