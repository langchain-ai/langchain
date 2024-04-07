from typing import Any, List, Optional, Type, TypedDict, cast

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import BaseGenerationOutputParser
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.pydantic_v1 import BaseModel


class _ToolCall(TypedDict):
    name: str
    args: dict
    id: str
    index: int


class ToolsOutputParser(BaseGenerationOutputParser):
    first_tool_only: bool = False
    args_only: bool = False
    pydantic_schemas: Optional[List[Type[BaseModel]]] = None

    class Config:
        extra = "forbid"

    def parse_result(self, result: List[Generation], *, partial: bool = False) -> Any:
        """Parse a list of candidate model Generations into a specific format.

        Args:
            result: A list of Generations to be parsed. The Generations are assumed
                to be different candidate outputs for a single model input.

        Returns:
            Structured output.
        """
        if not result or not isinstance(result[0], ChatGeneration):
            return None if self.first_tool_only else []
        tool_calls: List = _extract_tool_calls(result[0].message)
        if self.pydantic_schemas:
            tool_calls = [self._pydantic_parse(tc) for tc in tool_calls]
        elif self.args_only:
            tool_calls = [tc["args"] for tc in tool_calls]
        else:
            pass

        if self.first_tool_only:
            return tool_calls[0] if tool_calls else None
        else:
            return tool_calls

    def _pydantic_parse(self, tool_call: _ToolCall) -> BaseModel:
        cls_ = {schema.__name__: schema for schema in self.pydantic_schemas or []}[
            tool_call["name"]
        ]
        return cls_(**tool_call["args"])


def _extract_tool_calls(msg: BaseMessage) -> List[_ToolCall]:
    if isinstance(msg.content, str):
        return []
    tool_calls = []
    for i, block in enumerate(cast(List[dict], msg.content)):
        if block["type"] != "tool_use":
            continue
        tool_calls.append(
            _ToolCall(name=block["name"], args=block["input"], id=block["id"], index=i)
        )
    return tool_calls
