"""`Runnable` that routes to a set of `Runnable` objects."""

from __future__ import annotations

from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

from pydantic import ConfigDict
from typing_extensions import TypedDict, override

from langchain_core.runnables.base import (
    Runnable,
    RunnableSerializable,
    coerce_to_runnable,
)
from langchain_core.runnables.config import (
    RunnableConfig,
    get_config_list,
    get_executor_for_config,
)
from langchain_core.runnables.utils import (
    ConfigurableFieldSpec,
    Input,
    Output,
    gather_with_concurrency,
    get_unique_config_specs,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator


class RouterInput(TypedDict):
    """Router input."""

    key: str
    """The key to route on."""
    input: Any
    """The input to pass to the selected `Runnable`."""


class RouterRunnable(RunnableSerializable[RouterInput, Output]):
    """`Runnable` that routes to a set of `Runnable` based on `Input['key']`.

    Returns the output of the selected Runnable.

    Example:
        ```python
        from langchain_core.runnables.router import RouterRunnable
        from langchain_core.runnables import RunnableLambda

        add = RunnableLambda(func=lambda x: x + 1)
        square = RunnableLambda(func=lambda x: x**2)

        router = RouterRunnable(runnables={"add": add, "square": square})
        router.invoke({"key": "square", "input": 3})
        ```
    """

    runnables: Mapping[str, Runnable[Any, Output]]

    @property
    @override
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return get_unique_config_specs(
            spec for step in self.runnables.values() for spec in step.config_specs
        )

    def __init__(
        self,
        runnables: Mapping[str, Runnable[Any, Output] | Callable[[Any], Output]],
    ) -> None:
        """Create a `RouterRunnable`.

        Args:
            runnables: A mapping of keys to `Runnable` objects.
        """
        super().__init__(
            runnables={key: coerce_to_runnable(r) for key, r in runnables.items()}
        )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        """Return `True` as this class is serializable."""
        return True

    @classmethod
    @override
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "schema", "runnable"]`
        """
        return ["langchain", "schema", "runnable"]

    @override
    def invoke(
        self, input: RouterInput, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Output:
        key = input["key"]
        actual_input = input["input"]
        if key not in self.runnables:
            msg = f"No runnable associated with key '{key}'"
            raise ValueError(msg)

        runnable = self.runnables[key]
        return runnable.invoke(actual_input, config)

    @override
    async def ainvoke(
        self,
        input: RouterInput,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Output:
        key = input["key"]
        actual_input = input["input"]
        if key not in self.runnables:
            msg = f"No runnable associated with key '{key}'"
            raise ValueError(msg)

        runnable = self.runnables[key]
        return await runnable.ainvoke(actual_input, config)

    @override
    def batch(
        self,
        inputs: list[RouterInput],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> list[Output]:
        if not inputs:
            return []

        keys = [input_["key"] for input_ in inputs]
        actual_inputs = [input_["input"] for input_ in inputs]
        if any(key not in self.runnables for key in keys):
            msg = "One or more keys do not have a corresponding runnable"
            raise ValueError(msg)

        def invoke(
            runnable: Runnable[Input, Output], input_: Input, config: RunnableConfig
        ) -> Output | Exception:
            if return_exceptions:
                try:
                    return runnable.invoke(input_, config, **kwargs)
                except Exception as e:
                    return e
            else:
                return runnable.invoke(input_, config, **kwargs)

        runnables = [self.runnables[key] for key in keys]
        configs = get_config_list(config, len(inputs))
        with get_executor_for_config(configs[0]) as executor:
            return cast(
                "list[Output]",
                list(executor.map(invoke, runnables, actual_inputs, configs)),
            )

    @override
    async def abatch(
        self,
        inputs: list[RouterInput],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> list[Output]:
        if not inputs:
            return []

        keys = [input_["key"] for input_ in inputs]
        actual_inputs = [input_["input"] for input_ in inputs]
        if any(key not in self.runnables for key in keys):
            msg = "One or more keys do not have a corresponding runnable"
            raise ValueError(msg)

        async def ainvoke(
            runnable: Runnable[Input, Output], input_: Input, config: RunnableConfig
        ) -> Output | Exception:
            if return_exceptions:
                try:
                    return await runnable.ainvoke(input_, config, **kwargs)
                except Exception as e:
                    return e
            else:
                return await runnable.ainvoke(input_, config, **kwargs)

        runnables = [self.runnables[key] for key in keys]
        configs = get_config_list(config, len(inputs))
        return await gather_with_concurrency(
            configs[0].get("max_concurrency"),
            *map(ainvoke, runnables, actual_inputs, configs),
        )

    @override
    def stream(
        self,
        input: RouterInput,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[Output]:
        key = input["key"]
        actual_input = input["input"]
        if key not in self.runnables:
            msg = f"No runnable associated with key '{key}'"
            raise ValueError(msg)

        runnable = self.runnables[key]
        yield from runnable.stream(actual_input, config)

    @override
    async def astream(
        self,
        input: RouterInput,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> AsyncIterator[Output]:
        key = input["key"]
        actual_input = input["input"]
        if key not in self.runnables:
            msg = f"No runnable associated with key '{key}'"
            raise ValueError(msg)

        runnable = self.runnables[key]
        async for output in runnable.astream(actual_input, config):
            yield output
