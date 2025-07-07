from typing import Any, Optional, Union, cast

from langchain_core.messages import AIMessage, ToolCall
from langchain_core.messages.tool import tool_call
from langchain_core.output_parsers import BaseGenerationOutputParser
from langchain_core.outputs import ChatGeneration, Generation
from pydantic import BaseModel, ConfigDict


class ToolsOutputParser(BaseGenerationOutputParser):
    """Output parser for tool calls."""

    first_tool_only: bool = False
    """Whether to return only the first tool call."""
    args_only: bool = False
    """Whether to return only the arguments of the tool calls."""
    pydantic_schemas: Optional[list[type[BaseModel]]] = None
    """Pydantic schemas to parse tool calls into."""

    model_config = ConfigDict(
        extra="forbid",
    )

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """Parse a list of candidate model Generations into a specific format.

        Args:
            result: A list of Generations to be parsed. The Generations are assumed
                to be different candidate outputs for a single model input.
        Returns:
            Structured output.
        """
        if not result or not isinstance(result[0], ChatGeneration):
            return None if self.first_tool_only else []
        message = cast(AIMessage, result[0].message)
        tool_calls: list = [
            dict(tc) for tc in _extract_tool_calls_from_message(message)
        ]
        if isinstance(message.content, list):
            # Map tool call id to index
            id_to_index = {
                block["id"]: i
                for i, block in enumerate(message.content)
                if isinstance(block, dict) and block["type"] == "tool_use"
            }
            tool_calls = [{**tc, "index": id_to_index[tc["id"]]} for tc in tool_calls]
        if self.pydantic_schemas:
            tool_calls = [self._pydantic_parse(tc) for tc in tool_calls]
        elif self.args_only:
            tool_calls = [tc["args"] for tc in tool_calls]
        else:
            pass

        if self.first_tool_only:
            return tool_calls[0] if tool_calls else None
        else:
            return [tool_call for tool_call in tool_calls]

    def _pydantic_parse(self, tool_call: dict) -> BaseModel:
        cls_ = {schema.__name__: schema for schema in self.pydantic_schemas or []}[
            tool_call["name"]
        ]
        return cls_(**tool_call["args"])


def _extract_tool_calls_from_message(message: AIMessage) -> list[ToolCall]:
    """Extract tool calls from a list of content blocks."""
    if message.tool_calls:
        return message.tool_calls
    return extract_tool_calls(message.content)


def extract_tool_calls(content: Union[str, list[Union[str, dict]]]) -> list[ToolCall]:
    """Extract tool calls from a list of content blocks."""
    if isinstance(content, list):
        tool_calls = []
        for block in content:
            if isinstance(block, str):
                continue
            if block["type"] != "tool_use":
                continue
            tool_calls.append(
                tool_call(name=block["name"], args=block["input"], id=block["id"])
            )
        return tool_calls
    else:
        return []
