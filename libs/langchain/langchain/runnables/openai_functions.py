from collections.abc import Mapping
from operator import itemgetter
from typing import Any, Callable, Optional, Union

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.runnables import RouterRunnable, Runnable
from langchain_core.runnables.base import RunnableBindingBase
from typing_extensions import TypedDict


class OpenAIFunction(TypedDict):
    """A function description for ChatOpenAI"""

    name: str
    """The name of the function."""
    description: str
    """The description of the function."""
    parameters: dict
    """The parameters to the function."""


class OpenAIFunctionsRouter(RunnableBindingBase[BaseMessage, Any]):
    """A runnable that routes to the selected function."""

    functions: Optional[list[OpenAIFunction]]

    def __init__(
        self,
        runnables: Mapping[
            str,
            Union[
                Runnable[dict, Any],
                Callable[[dict], Any],
            ],
        ],
        functions: Optional[list[OpenAIFunction]] = None,
    ):
        if functions is not None:
            if len(functions) != len(runnables):
                msg = "The number of functions does not match the number of runnables."
                raise ValueError(msg)
            if not all(func["name"] in runnables for func in functions):
                msg = "One or more function names are not found in runnables."
                raise ValueError(msg)
        router = (
            JsonOutputFunctionsParser(args_only=False)
            | {"key": itemgetter("name"), "input": itemgetter("arguments")}
            | RouterRunnable(runnables)
        )
        super().__init__(bound=router, kwargs={}, functions=functions)
