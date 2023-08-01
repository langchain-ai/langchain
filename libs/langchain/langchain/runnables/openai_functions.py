from operator import itemgetter
from typing import Any, Callable, List, TypedDict, Union
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.schema.output import ChatGeneration
from langchain.schema.runnable import RouterRunnable, Runnable, RunnableBinding


class OpenAIFunction(TypedDict):
    """A function to call on the OpenAI API."""

    name: str
    """The name of the function."""
    description: str
    """The description of the function."""
    parameters: dict
    """The parameters to the function."""


class OpenAIRunnableFunction(OpenAIFunction):
    """A function to call on the OpenAI API."""

    runnable: Union[Runnable[dict, Any], Callable[[dict], Any]]
    """The runnable to call."""


class OpenAIFunctionsRouter(RunnableBinding[ChatGeneration, Any]):
    """A runnable that routes to the selected function."""

    functions: List[OpenAIFunction]

    def __init__(self, functions: List[OpenAIFunction]):
        functions = [func.copy() for func in functions]
        router = (
            JsonOutputFunctionsParser(args_only=False)
            | {"key": itemgetter("name"), "input": itemgetter("arguments")}
            | RouterRunnable({func["name"]: func.pop("runnable") for func in functions})
        )
        super().__init__(bound=router, kwargs={}, functions=functions)
