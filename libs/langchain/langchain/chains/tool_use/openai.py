import json
from operator import itemgetter
from typing import Any, Optional, Sequence

from langchain_community.tools.convert_to_openai import (
    format_tool_to_openai_function,
    format_tool_to_openai_tool,
)
from langchain_core.language_models import LanguageModelInput, LanguageModelOutput
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import BasePromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough,
)
from langchain_core.tools import Tool

from langchain.output_parsers import JsonOutputToolsParser
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser


def _call_tool(
    tool_api_output: list, tools: Sequence[Tool], config: RunnableConfig
) -> Runnable:
    tool_map = {tool.name: tool for tool in tools}
    chosen = {}
    for i, tool_call in enumerate(tool_api_output):
        chosen[tool_call["type"]] = (
            RunnableLambda(itemgetter(i))
            | RunnableLambda(itemgetter("args"))
            | tool_map[tool_call["type"]]
        )
    return RunnableMap(chosen)


class _ToolException(Exception):
    """LangChain tool exception."""

    def __init__(self, tool_calls: list, base_exception: Exception) -> None:
        super().__init__()
        self.tool_calls = tool_calls
        self.base_exception = base_exception


def call_tool(
    tool_api_output: list, tools: Sequence[Tool], config: RunnableConfig
) -> Any:
    tool_map = {tool.name: tool for tool in tools}
    chosen = {}
    for i, tool_call in enumerate(tool_api_output):
        chosen[tool_call["type"]] = (
            RunnableLambda(itemgetter(i))
            | RunnableLambda(itemgetter("args"))
            | tool_map[tool_call["type"]]
        )
    try:
        return chosen.invoke(tool_api_output, config=config)
    except Exception as e:
        return _ToolException(
            tool_calls=tool_api_output,
            base_exception=e,
        )


def retry_if_exception(inputs: dict, chain: Runnable, config: RunnableConfig) -> Any:
    last_output = inputs["last_output"]
    if isinstance(last_output, _ToolException):
        tool_calls = [
            {
                "type": "function",
                "function": {
                    "name": tool_call["type"],
                    "arguments": json.dumps(tool_call["arguments"]),
                },
            }
            for tool_call in last_output.tool_calls
        ]
        messages = [
            AIMessage(content="", additional_kwargs={"tool_calls": tool_calls}),
            HumanMessage(
                content=f"The last tool calls raised exception:\n\n{last_output.base_exception}\n\nTry calling the tools again with corrected arguments."
            ),
        ]
        return RunnablePassthrough.assign(last_output=lambda x: messages) | chain
    else:
        return inputs["last_output"]


def create_openai_tools_chain(
    llm: Runnable[LanguageModelInput, LanguageModelOutput],
    tools: Sequence[Tool],
    *,
    prompt: Optional[BasePromptTemplate] = None,
    enforce_tool_usage: bool = False,
    self_correct: bool = False,
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
        if self_correct and "last_output" not in prompt.input_variables:
            prompt = prompt + MessagesPlaceholder("last_output", optional=True)
        chain = prompt | chain
    if self_correct:
        if not prompt:
            raise ValueError
        chain = RunnablePassthrough.assign(last_output=chain) | RunnableLambda(
            retry_if_exception
        ).bind(chain=chain)
    return chain


def _call_tool_from_function(
    function_api_output: dict, tools: Sequence[Tool]
) -> Runnable:
    tool_map = {tool.name: tool for tool in tools}
    chosen = tool_map[function_api_output["name"]]
    return itemgetter("arguments") | chosen


def create_openai_functions_chain(
    llm: Runnable[LanguageModelInput, LanguageModelOutput],
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
    chain: Runnable = (
        llm_with_functions | JsonOutputFunctionsParser(args_only=False) | call_tool
    )
    if prompt:
        chain = prompt | chain
    return chain
