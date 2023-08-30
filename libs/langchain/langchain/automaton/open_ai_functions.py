from __future__ import annotations

import json
from typing import List, Any, TypedDict, Sequence, Optional

from langchain.base_language import BaseLanguageModel
from langchain.prompts.chat import ChatPromptTemplate
from langchain.runnables.openai_functions import OpenAIFunctionsRouter
from langchain.schema import BaseMessage, AIMessage
from langchain.schema.prompt import PromptValue
from langchain.schema.runnable import Runnable
from langchain.tools.base import BaseTool
from langchain.tools.convert_to_openai import format_tool_to_openai_function


class FunctionCall(TypedDict):
    name: str
    """The name of the function."""
    arguments: dict
    """The arguments to the function."""
    result: Any  # Need to denote not invoked yet as well
    """The result of the function call"""


class ActingResult(TypedDict):
    """The result of an action."""

    message: BaseMessage
    """The message that was passed to the action."""
    function_call: Optional[FunctionCall]


def create_action_taking_llm(
    llm: BaseLanguageModel,
    *,
    tools: Sequence[BaseTool] = (),
    stop: Sequence[str] | None = None,
    invoke_function: bool = True,
) -> Runnable:
    """A chain that can create an action.

    Args:
        llm: The language model to use.
        tools: The tools to use.
        stop: The stop tokens to use.
        invoke_function: Whether to invoke the function.

    Returns:
        a segment of a runnable that take an action.
    """

    openai_funcs = [format_tool_to_openai_function(tool_) for tool_ in tools]

    def _interpret_message(message: BaseMessage) -> ActingResult:
        """Interpret a message."""
        if (
            isinstance(message, AIMessage)
            and "function_call" in message.additional_kwargs
        ):
            if invoke_function:
                result = invoke_from_function.invoke(  # TODO: fixme using invoke
                    message
                )
            else:
                result = None
            return {
                "message": message,
                "function_call": {
                    "name": message.additional_kwargs["function_call"]["name"],
                    "arguments": json.loads(
                        message.additional_kwargs["function_call"]["arguments"]
                    ),
                    "result": result,
                    # Check this works.
                    # "result": message.additional_kwargs["function_call"]
                    #           | invoke_from_function,
                },
            }
        else:
            return {
                "message": message,
                "function_call": None,
            }

    invoke_from_function = OpenAIFunctionsRouter(
        functions=openai_funcs,
        runnables={
            openai_func["name"]: tool_
            for openai_func, tool_ in zip(openai_funcs, tools)
        },
    )

    if stop:
        _llm = llm.bind(stop=stop)
    else:
        _llm = llm

    chain = _llm.bind(functions=openai_funcs) | _interpret_message
    return chain


def _interpret_message(message: BaseMessage) -> ActingResult:
    """Interpret a message."""
    if isinstance(message, AIMessage) and "function_call" in message.additional_kwargs:
        raise NotImplementedError()
        # if invoke_function:
        #     result = invoke_from_function.invoke(message)  # TODO: fixme using invoke
        # else:
        #     result = None
        # return {
        #     "message": message,
        #     "function_call": {
        #         "name": message.additional_kwargs["function_call"]["name"],
        #         "arguments": json.loads(
        #             message.additional_kwargs["function_call"]["arguments"]
        #         ),
        #         "result": result,
        #         # Check this works.
        #         # "result": message.additional_kwargs["function_call"]
        #         #           | invoke_from_function,
        #     },
        # }
    else:
        return {
            "message": message,
            "function_call": None,
        }


def mrkl_parse(expect: str, message: str) -> Optional[str]:
    """"""
    if expect == "Thought: ":
        if message.startswith("Thought: "):
            return message
    elif expect == "Observation: ":
        if message.startswith("Observation: "):
            return message
    else:
        return None


def create_action_taking_llm_2(
    llm: BaseLanguageModel,
    *,
    tools: Sequence[BaseTool] = (),
    stop: Sequence[str] | None = None,
    parser: Optional[Runnable] = None,
) -> Runnable:
    """A chain that can create an action.

    Args:
        llm: The language model to use.
        tools: The tools to use.
        stop: The stop tokens to use.
        invoke_function: Whether to invoke the function.

    Returns:
        a segment of a runnable that take an action.
    """

    if stop:
        _llm = llm.bind(stop=stop)
    else:
        _llm = llm

    if parser:
        chain = _llm | parser
    else:
        chain = _llm
    return chain


class SimpleChatGenerator(PromptValue):
    """Base abstract class for inputs to any language model.

    PromptValues can be converted to both LLM (pure text-generation) inputs and
        ChatModel inputs.
    """

    messages: Sequence[BaseMessage] | ChatPromptTemplate

    @property
    def lc_serializable(self) -> bool:
        """Return whether or not the class is serializable."""
        return True

    def to_string(self) -> str:
        """Return prompt value as string."""
        finalized = []
        for message in self.to_messages():
            prefix = message.type
            finalized.append(f"{prefix}: {message.content}")
        return "\n".join(finalized) + "\n" + "ai:"

    def to_messages(self) -> List[BaseMessage]:
        """Return prompt as a list of Messages."""
        if isinstance(self.messages, ChatPromptTemplate):
            return self.messages.format_messages()
        return list(self.messages)
