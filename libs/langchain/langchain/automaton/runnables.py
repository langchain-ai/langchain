"""Module contains useful runnables for agents."""
from __future__ import annotations

from typing import Sequence, Callable, List, Optional, Union

from langchain.automaton.typedefs import (
    MessageLike,
    MessageLog,
    FunctionResult,
    FunctionCall,
)
from langchain.schema import BaseMessage, AIMessage, PromptValue
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.runnable.base import RunnableLambda, Runnable
from langchain.tools import BaseTool
from langchain.tools.convert_to_openai import format_tool_to_openai_function


# PUBLIC API


def create_tool_invoker(
    tools: Sequence[BaseTool],
) -> Runnable[FunctionCall, FunctionResult]:
    """Re-write with router."""
    tools_by_name = {tool.name: tool for tool in tools}

    def func(function_call: FunctionCall) -> FunctionResult:
        """A function that can invoke a tool using .run"""
        try:
            tool = tools_by_name[function_call.name]
        except KeyError:
            raise AssertionError(f"No such tool: {function_call.name}")
        try:
            result = tool.run(function_call.arguments or {})
            error = None
        except Exception as e:
            result = None
            error = repr(e) + repr(function_call.arguments)
        return FunctionResult(name=function_call.name, result=result, error=error)

    return RunnableLambda(func=func)


def create_llm_program(
    llm: BaseLanguageModel,
    prompt_generator: Callable[[MessageLog], PromptValue],
    *,
    tools: Optional[Sequence[BaseTool]] = None,
    stop: Optional[Sequence[str]] = None,
    parser: Union[
        Runnable[Union[BaseMessage, str], MessageLike],
        Callable[[Union[BaseMessage, str]], MessageLike],
        None,
    ] = None,
    invoke_tools: bool = True,
) -> Runnable[MessageLog, List[MessageLike]]:
    """Create a runnable that can update memory."""

    tool_invoker = create_tool_invoker(tools) if invoke_tools else None
    openai_funcs = [format_tool_to_openai_function(tool_) for tool_ in tools]

    def _bound(message_log: MessageLog):
        messages = []
        prompt_value = prompt_generator(message_log)
        llm_chain = llm
        if stop:
            llm_chain = llm_chain.bind(stop=stop)
        if tools:
            llm_chain = llm_chain.bind(tools=openai_funcs)

        result = llm_chain.invoke(prompt_value)

        if isinstance(result, BaseMessage):
            messages.append(result)
        elif isinstance(result, str):
            messages.append(AIMessage(content=result))
        else:
            raise NotImplementedError(f"Unsupported type {type(result)}")

        if parser:
            if not isinstance(parser, Runnable):
                _parser = RunnableLambda(parser)
            else:
                _parser = parser
            parsed_result = _parser.invoke(result)
            if parsed_result:
                if not isinstance(parsed_result, MessageLike):
                    raise TypeError(
                        f"Expected a MessageLike type got: {type(parsed_result)}"
                    )
                messages.append(parsed_result)

        last_message = messages[-1]

        if tool_invoker and isinstance(last_message, FunctionCall):
            function_result = tool_invoker.invoke(last_message)
            messages.append(function_result)

        return messages

    return RunnableLambda(
        func=_bound,
    )
