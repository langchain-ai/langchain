from __future__ import annotations

from operator import itemgetter
from typing import Any, Callable, List, Mapping, TypedDict, Union, Sequence, Optional

from langchain.base_language import BaseLanguageModel
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.schema import BaseMessage, AIMessage
from langchain.schema.output import ChatGeneration
from langchain.schema.runnable import RouterRunnable, Runnable, RunnableBinding
from langchain.tools.base import BaseTool
from langchain.tools.convert_to_openai import format_tool_to_openai_function


class OpenAIFunction(TypedDict):
    """A function to call on the OpenAI API."""

    name: str
    """The name of the function."""
    description: str
    """The description of the function."""
    parameters: dict
    """The parameters to the function."""


class OpenAIFunctionsRouter(RunnableBinding[ChatGeneration, Any]):
    """A runnable that routes to the selected function."""

    functions: List[OpenAIFunction]

    def __init__(
        self,
        functions: List[OpenAIFunction],
        runnables: Mapping[
            str,
            Union[
                Runnable[dict, Any],
                Callable[[dict], Any],
            ],
        ],
    ):
        assert len(functions) == len(runnables)
        assert all(func["name"] in runnables for func in functions)
        router = (
            JsonOutputFunctionsParser(args_only=False)
            | {"key": itemgetter("name"), "input": itemgetter("arguments")}
            | RouterRunnable(runnables)
        )
        super().__init__(bound=router, kwargs={}, functions=functions)


class FunctionCallRequest(TypedDict):
    name: str
    """The name of the function."""
    arguments: dict
    """The arguments to the function."""


class FunctionCall(TypedDict):
    request: FunctionCallRequest
    """The function call request."""
    result: Any
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
) -> Runnable:
    """A chain that can create an action.

    Args:
        llm: The language model to use.
        tools: The tools to use.
        stop: The stop tokens to use.

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
            return {
                "message": message,
                "function_call": {
                    "request": {
                        "name": message.additional_kwargs["function_call"]["name"],
                        "arguments": {"fixme": "fixme"},
                    },
                    "result": invoke_from_function.invoke(  # TODO: fixme using invoke
                        message.additional_kwargs["function_call"]
                    ),
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
