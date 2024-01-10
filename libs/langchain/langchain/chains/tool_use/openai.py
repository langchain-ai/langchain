from operator import itemgetter
from typing import Optional, Sequence

from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_community.tools.convert_to_openai import (
    format_tool_to_openai_function,
    format_tool_to_openai_tool,
)
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnableMap
from langchain_core.tools import Tool

from langchain.output_parsers import JsonOutputToolsParser


def _call_tool(tool_api_output: list, tools: Sequence[Tool]) -> Runnable:
    tool_map = {tool.name: tool for tool in tools}
    chosen = {}
    for i, tool_call in enumerate(tool_api_output):
        chosen[tool_call["type"]] = RunnableLambda(itemgetter(i)) | RunnableLambda(itemgetter("args")) | tool_map[tool_call["type"]]
    return RunnableMap(chosen)


def create_openai_tools_chain(
    llm: BaseLanguageModel,
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
    if enforce_tool_usage:
        chain = chain | itemgetter(tools[0].name)
    return chain


def _call_tool_from_function(
    function_api_output: dict, tools: Sequence[Tool]
) -> Runnable:
    tool_map = {tool.name: tool for tool in tools}
    chosen = tool_map[function_api_output["name"]]
    return itemgetter("arguments") | chosen


def create_openai_functions_chain(
    llm: BaseLanguageModel,
    tools: Sequence[Tool],
    *,
    prompt: Optional[BasePromptTemplate] = None,
    enforce_tool_usage: bool = False,
) -> Runnable:
    """"""
    if enforce_tool_usage and len(tools) > 1:
        raise ValueError()

    llm_with_functions = llm.bind(
        functions=[format_tool_to_openai_function(t) for t in tools]
    )
    if enforce_tool_usage:
        llm_with_functions = llm_with_functions.bind(
            function_call={"name": tools[0].name}
        )
    call_tool: Runnable = RunnableLambda(_call_tool_from_function).bind(tools=tools)
    chain: Runnable = llm_with_functions | JsonOutputFunctionsParser(args_only=False) | call_tool
    if prompt:
        chain = prompt | chain
    return chain
