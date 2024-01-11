import copy
import json
from json import JSONDecodeError
from typing import Any, List, Type

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import (
    BaseGenerationOutputParser,
)
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.pydantic_v1 import BaseModel


class JsonOutputToolsParser(BaseGenerationOutputParser[Any]):
    """Parse tools from OpenAI response."""

    strict: bool = False
    """Whether to allow non-JSON-compliant strings.

    See: https://docs.python.org/3/library/json.html#encoders-and-decoders

    Useful when the parsed output may include unicode characters or new lines.
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
                pass
            try:
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
            final_tools.append(
                {
                    "type": tool_call["function"]["name"],
                    "args": function_args,
                }
            )
        if exceptions:
            raise OutputParserException("\n\n".join(exceptions))
        return final_tools


class JsonOutputKeyToolsParser(JsonOutputToolsParser):
    """Parse tools from OpenAI response."""

    key_name: str
    """The type of tools to return."""

    def parse_result(self, result: List[Generation], *, partial: bool = False) -> Any:
        results = super().parse_result(result)
        return [res["args"] for res in results if results["type"] == self.key_name]


class PydanticToolsParser(JsonOutputToolsParser):
    """Parse tools from OpenAI response."""

    tools: List[Type[BaseModel]]

    def parse_result(self, result: List[Generation], *, partial: bool = False) -> Any:
        results = super().parse_result(result)
        name_dict = {tool.__name__: tool for tool in self.tools}
        return [name_dict[res["type"]](**res["args"]) for res in results]
