"""Module contains well known runnables for agents."""
from __future__ import annotations

from typing import Sequence, Callable, Union, List, Optional

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



def create_tool_invoker(
    tools: Sequence[BaseTool],
) -> Runnable[FunctionCall, FunctionResult]:
    """Re-write with router."""
    tools_by_name = {tool.name: tool for tool in tools}

    def func(input: FunctionCall) -> FunctionResult:
        """A function that can invoke a tool using .run"""
        tool = tools_by_name[input.name]
        try:
            result = tool.run(input.arguments)
            error = None
        except Exception as e:
            result = None
            error = repr(e)
        return FunctionResult(result=result, error=error)

    return RunnableLambda(func=func)


def create_llm_program(
    llm: BaseLanguageModel,
    prompt_generator: Callable[[MessageLog], PromptValue],
    *,
    stop: Optional[Sequence[str]] = None,
    parser: Union[Runnable, Callable] = None,
) -> Runnable[MessageLog, List[MessageLike]]:
    """Create a runnable that can update memory."""

    def _bound(event_log: MessageLog):
        messages = []
        prompt_value = prompt_generator(event_log)
        llm_chain = llm
        if stop:
            llm_chain = llm_chain.bind(stop=stop)

        result = llm_chain.invoke(prompt_value)

        if isinstance(result, BaseMessage):
            messages.append(result)
        elif isinstance(result, str):
            messages.append(AIMessage(content=result))
        else:
            raise NotImplementedError(f"Unsupported type {result}")

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

        return messages

    return RunnableLambda(
        func=_bound,
    )
