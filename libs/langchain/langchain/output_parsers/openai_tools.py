import copy
import json
from typing import Any, List, Type

from langchain.pydantic_v1 import BaseModel
from langchain.schema import (
    ChatGeneration,
    Generation,
    OutputParserException,
)
from langchain.schema.output_parser import (
    BaseGenerationOutputParser,
)


class JsonOutputToolsParser(BaseGenerationOutputParser[Any]):
    """Parse tools from OpenAI response."""

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
        for tool_call in tool_calls:
            if "function" not in tool_call:
                pass
            function_args = tool_call["function"]["arguments"]
            final_tools.append(
                {
                    "type": tool_call["function"]["name"],
                    "args": json.loads(function_args),
                }
            )
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
