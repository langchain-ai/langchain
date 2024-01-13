from operator import itemgetter
from typing import Optional, Sequence

from langchain_community.tools.convert_to_openai import (
    format_tool_to_openai_tool,
)
from langchain_core.language_models import LanguageModelInput, LanguageModelOutput
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnableMap
from langchain_core.tools import Tool

from langchain.output_parsers import JsonOutputToolsParser


def _call_tool(tool_api_output: list, tools: Sequence[Tool]) -> Runnable:
    tool_map = {tool.name: tool for tool in tools}
    chosen = {}
    for i, tool_call in enumerate(tool_api_output):
        chosen[tool_call["type"]] = (
            RunnableLambda(itemgetter(i))
            | itemgetter("args")
            | tool_map[tool_call["type"]]
        )
    return RunnableMap(chosen)


def create_openai_tools_chain(
    llm: Runnable[LanguageModelInput, LanguageModelOutput],
    tools: Sequence[Tool],
    *,
    prompt: Optional[BasePromptTemplate] = None,
    enforce_tool_usage: bool = False,
) -> Runnable:
    """"""
    if enforce_tool_usage and len(tools) > 1:
        raise ValueError("Cannot enforce tool usage with more than one tool.")
    llm_with_tools = llm.bind(tools=[format_tool_to_openai_tool(t) for t in tools])
    if enforce_tool_usage:
        llm_with_tools = llm_with_tools.bind(
            tool_choice={"type": "function", "function": {"name": tools[0].name}}
        )
    call_tool: Runnable = RunnableLambda(_call_tool).bind(tools=tools)
    chain: Runnable = llm_with_tools | JsonOutputToolsParser() | call_tool
    if prompt:
        chain = prompt | chain
    return chain
