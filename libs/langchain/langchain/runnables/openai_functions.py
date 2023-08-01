from operator import itemgetter
from typing import Any, Callable, List, Mapping, TypedDict, Union
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
