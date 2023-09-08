from __future__ import annotations

from typing import (
    Any,
    AsyncIterator,
    Callable,
    Generic,
    Iterator,
    List,
    Mapping,
    Optional,
    TypedDict,
    Union,
    cast,
)

from langchain.load.serializable import Serializable
from langchain.schema.runnable.base import (
    Input,
    Other,
    Output,
    Runnable,
    RunnableSequence,
    coerce_to_runnable,
)
from langchain.schema.runnable.config import (
    RunnableConfig,
    get_config_list,
    get_executor_for_config,
)
from langchain.schema.runnable.utils import gather_with_concurrency


class RouterInput(TypedDict):
    """A Router input.

    Attributes:
        key: The key to route on.
        input: The input to pass to the selected runnable.
    """

    key: str
    input: Any


class RouterRunnable(
    Serializable, Generic[Input, Output], Runnable[RouterInput, Output]
):
    """
    A runnable that routes to a set of runnables based on Input['key'].
    Returns the output of the selected runnable.
    """

    runnables: Mapping[str, Runnable[Input, Output]]

    def __init__(
        self,
        runnables: Mapping[
            str, Union[Runnable[Input, Output], Callable[[Input], Output]]
        ],
    ) -> None:
        super().__init__(
            runnables={key: coerce_to_runnable(r) for key, r in runnables.items()}
        )

    class Config:
        arbitrary_types_allowed = True

    @property
    def lc_serializable(self) -> bool:
        return True

    @property
    def lc_namespace(self) -> List[str]:
        return self.__class__.__module__.split(".")[:-1]

    def __or__(
        self,
        other: Union[
            Runnable[Any, Other],
            Callable[[Any], Other],
            Mapping[str, Union[Runnable[Any, Other], Callable[[Any], Other]]],
            Mapping[str, Any],
        ],
    ) -> RunnableSequence[RouterInput, Other]:
        return RunnableSequence(first=self, last=coerce_to_runnable(other))

    def __ror__(
        self,
        other: Union[
            Runnable[Other, Any],
            Callable[[Any], Other],
            Mapping[str, Union[Runnable[Other, Any], Callable[[Other], Any]]],
            Mapping[str, Any],
        ],
    ) -> RunnableSequence[Other, Output]:
        return RunnableSequence(first=coerce_to_runnable(other), last=self)

    def invoke(
        self, input: RouterInput, config: Optional[RunnableConfig] = None
    ) -> Output:
        key = input["key"]
        actual_input = input["input"]
        if key not in self.runnables:
            raise ValueError(f"No runnable associated with key '{key}'")

        runnable = self.runnables[key]
        return runnable.invoke(actual_input, config)

    async def ainvoke(
        self,
        input: RouterInput,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Output:
        key = input["key"]
        actual_input = input["input"]
        if key not in self.runnables:
            raise ValueError(f"No runnable associated with key '{key}'")

        runnable = self.runnables[key]
        return await runnable.ainvoke(actual_input, config)

    def batch(
        self,
        inputs: List[RouterInput],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> List[Output]:
        keys = [input["key"] for input in inputs]
        actual_inputs = [input["input"] for input in inputs]
        if any(key not in self.runnables for key in keys):
            raise ValueError("One or more keys do not have a corresponding runnable")

        def invoke(
            runnable: Runnable, input: Input, config: RunnableConfig
        ) -> Union[Output, Exception]:
            if return_exceptions:
                try:
                    return runnable.invoke(input, config, **kwargs)
                except Exception as e:
                    return e
            else:
                return runnable.invoke(input, config, **kwargs)

        runnables = [self.runnables[key] for key in keys]
        configs = get_config_list(config, len(inputs))
        with get_executor_for_config(configs[0]) as executor:
            return cast(
                List[Output],
                list(executor.map(invoke, runnables, actual_inputs, configs)),
            )

    async def abatch(
        self,
        inputs: List[RouterInput],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> List[Output]:
        keys = [input["key"] for input in inputs]
        actual_inputs = [input["input"] for input in inputs]
        if any(key not in self.runnables for key in keys):
            raise ValueError("One or more keys do not have a corresponding runnable")

        async def ainvoke(
            runnable: Runnable, input: Input, config: RunnableConfig
        ) -> Union[Output, Exception]:
            if return_exceptions:
                try:
                    return await runnable.ainvoke(input, config, **kwargs)
                except Exception as e:
                    return e
            else:
                return await runnable.ainvoke(input, config, **kwargs)

        runnables = [self.runnables[key] for key in keys]
        configs = get_config_list(config, len(inputs))
        return await gather_with_concurrency(
            configs[0].get("max_concurrency"),
            *(
                ainvoke(runnable, input, config)
                for runnable, input, config in zip(runnables, actual_inputs, configs)
            ),
        )

    def stream(
        self,
        input: RouterInput,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Output]:
        key = input["key"]
        actual_input = input["input"]
        if key not in self.runnables:
            raise ValueError(f"No runnable associated with key '{key}'")

        runnable = self.runnables[key]
        yield from runnable.stream(actual_input, config)

    async def astream(
        self,
        input: RouterInput,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Output]:
        key = input["key"]
        actual_input = input["input"]
        if key not in self.runnables:
            raise ValueError(f"No runnable associated with key '{key}'")

        runnable = self.runnables[key]
        async for output in runnable.astream(actual_input, config):
            yield output
