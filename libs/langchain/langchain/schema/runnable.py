import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Any,
    AsyncIterator,
    Coroutine,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    TypedDict,
    TypeVar,
    Union,
    cast,
)

from pydantic import Field

from langchain.callbacks.base import BaseCallbackManager, Callbacks
from langchain.load.dump import dumpd
from langchain.load.serializable import Serializable


async def _gated_coro(semaphore: asyncio.Semaphore, coro: Coroutine) -> Any:
    async with semaphore:
        return await coro


async def _gather_with_concurrency(n: Union[int, None], *coros: Coroutine) -> list:
    if n is None:
        return await asyncio.gather(*coros)

    semaphore = asyncio.Semaphore(n)

    return await asyncio.gather(*(_gated_coro(semaphore, c) for c in coros))


class RunnableConfig(TypedDict, total=False):
    """
    Tags for this call and any sub-calls (eg. a Chain calling an LLM).
    You can use these to filter calls.
    """

    tags: List[str]

    """
    Metadata for this call and any sub-calls (eg. a Chain calling an LLM).
    Keys should be strings, values should be JSON-serializable.
    """
    metadata: Dict[str, Any]

    """
    Callbacks for this call and any sub-calls (eg. a Chain calling an LLM).
    Tags are passed to all callbacks, metadata is passed to handle*Start callbacks.
    """
    callbacks: Callbacks


Input = TypeVar("Input")
# Output type should implement __concat__, as eg str, list, dict do
Output = TypeVar("Output")
Other = TypeVar("Other")


class Runnable(Generic[Input, Output], ABC):
    def __or__(
        self, __value: "Runnable[Any, Other]"
    ) -> "RunnableSequence[Input, Other]":
        return RunnableSequence(first=self, last=__value)

    @abstractmethod
    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        ...

    async def ainvoke(
        self, input: Input, config: Optional[RunnableConfig] = None
    ) -> Output:
        return await asyncio.get_running_loop().run_in_executor(
            None, self.invoke, input, config
        )

    def batch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        max_concurrency: Optional[int] = None,
    ) -> List[Output]:
        configs = self._get_config_list(config, len(inputs))

        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            futures = [
                executor.submit(self.invoke, input, config)
                for input, config in zip(inputs, configs)
            ]
            return [future.result() for future in futures]

    async def abatch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        max_concurrency: Optional[int] = None,
    ) -> List[Output]:
        configs = self._get_config_list(config, len(inputs))
        coros = (self.ainvoke(input, config) for input, config in zip(inputs, configs))

        return await _gather_with_concurrency(max_concurrency, *coros)

    def stream(
        self, input: Input, config: Optional[RunnableConfig] = None
    ) -> Iterator[Output]:
        yield self.invoke(input, config)

    async def astream(
        self, input: Input, config: Optional[RunnableConfig] = None
    ) -> AsyncIterator[Output]:
        yield await self.ainvoke(input, config)

    def _get_config_list(
        self, config: Optional[Union[RunnableConfig, List[RunnableConfig]]], length: int
    ) -> List[RunnableConfig]:
        if isinstance(config, list) and len(config) != length:
            raise ValueError(
                f"config must be a list of the same length as inputs, "
                f"but got {len(config)} configs for {length} inputs"
            )

        return config if isinstance(config, list) else [config or {}] * length


class RunnableSequence(Serializable, Runnable[Input, Output]):
    first: Runnable[Input, Any]
    middle: List[Runnable[Any, Any]] = Field(default_factory=list)
    last: Runnable[Any, Output]

    class Config:
        arbitrary_types_allowed = True

    def __or__(self, __value: Runnable[Any, Other]) -> "RunnableSequence[Input, Other]":
        if isinstance(__value, RunnableSequence):
            return RunnableSequence(
                first=self.first,
                middle=self.middle + [self.last] + __value.middle,
                last=__value.last,
            )
        else:
            return RunnableSequence(
                first=self.first, middle=self.middle + [self.last], last=__value
            )

    def __ror__(
        self, __value: Runnable[Other, Any]
    ) -> "RunnableSequence[Other, Output]":
        if isinstance(__value, RunnableSequence):
            return RunnableSequence(
                first=__value.first,
                middle=__value.middle + [__value.last] + self.middle,
                last=self.last,
            )
        else:
            return RunnableSequence(
                first=__value, middle=[self.first] + self.middle, last=self.last
            )

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        from langchain.callbacks.manager import CallbackManager

        # setup callbacks
        config = config or {}
        cm = CallbackManager.configure(
            inheritable_callbacks=config.get("callbacks"),
            local_callbacks=None,
            verbose=False,
            inheritable_tags=config.get("tags"),
            local_tags=None,
            inheritable_metadata=config.get("metadata"),
            local_metadata=None,
        )
        rm = cm.on_chain_start(dumpd(self), {"input": input})

        # invoke
        try:
            for step in [self.first] + self.middle + [self.last]:
                input = step.invoke(input, _patch_config(config, rm.get_child()))
        except (KeyboardInterrupt, Exception) as e:
            rm.on_chain_error(e)
            raise
        else:
            rm.on_chain_end({"output": input})
            return cast(Output, input)

    async def ainvoke(
        self, input: Input, config: Optional[RunnableConfig] = None
    ) -> Output:
        from langchain.callbacks.manager import AsyncCallbackManager

        # setup callbacks
        config = config or {}
        cm = AsyncCallbackManager.configure(
            inheritable_callbacks=config.get("callbacks"),
            local_callbacks=None,
            verbose=False,
            inheritable_tags=config.get("tags"),
            local_tags=None,
            inheritable_metadata=config.get("metadata"),
            local_metadata=None,
        )
        rm = await cm.on_chain_start(dumpd(self), {"input": input})

        # invoke
        try:
            for step in [self.first] + self.middle + [self.last]:
                input = await step.ainvoke(input, _patch_config(config, rm.get_child()))
        except (KeyboardInterrupt, Exception) as e:
            await rm.on_chain_error(e)
            raise
        else:
            await rm.on_chain_end({"output": input})
            return cast(Output, input)

    def batch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        max_concurrency: Optional[int] = None,
    ) -> List[Output]:
        from langchain.callbacks.manager import CallbackManager

        # setup callbacks
        configs = self._get_config_list(config, len(inputs))
        cms = [
            CallbackManager.configure(
                inheritable_callbacks=config.get("callbacks"),
                local_callbacks=None,
                verbose=False,
                inheritable_tags=config.get("tags"),
                local_tags=None,
                inheritable_metadata=config.get("metadata"),
                local_metadata=None,
            )
            for config in configs
        ]
        rms = [
            cm.on_chain_start(dumpd(self), {"input": input})
            for cm, input in zip(cms, inputs)
        ]

        # invoke
        try:
            for step in [self.first] + self.middle + [self.last]:
                inputs = step.batch(
                    inputs,
                    [
                        _patch_config(config, rm.get_child())
                        for rm, config in zip(rms, configs)
                    ],
                    max_concurrency=max_concurrency,
                )
        except (KeyboardInterrupt, Exception) as e:
            for rm in rms:
                rm.on_chain_error(e)
            raise
        else:
            for rm, input in zip(rms, inputs):
                rm.on_chain_end({"output": input})
            return cast(List[Output], inputs)

    async def abatch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        max_concurrency: Optional[int] = None,
    ) -> List[Output]:
        from langchain.callbacks.manager import (
            AsyncCallbackManager,
            AsyncCallbackManagerForChainRun,
        )

        # setup callbacks
        configs = self._get_config_list(config, len(inputs))
        cms = [
            AsyncCallbackManager.configure(
                inheritable_callbacks=config.get("callbacks"),
                local_callbacks=None,
                verbose=False,
                inheritable_tags=config.get("tags"),
                local_tags=None,
                inheritable_metadata=config.get("metadata"),
                local_metadata=None,
            )
            for config in configs
        ]
        rms: List[AsyncCallbackManagerForChainRun] = await asyncio.gather(
            *[
                cm.on_chain_start(dumpd(self), {"input": input})
                for cm, input in zip(cms, inputs)
            ]
        )

        # invoke batch on each step
        # this uses batching optimizations in subclasses, like LLM
        try:
            for step in [self.first] + self.middle + [self.last]:
                inputs = await step.abatch(
                    inputs,
                    [
                        _patch_config(config, rm.get_child())
                        for rm, config in zip(rms, configs)
                    ],
                    max_concurrency=max_concurrency,
                )
        except (KeyboardInterrupt, Exception) as e:
            await asyncio.gather(*[rm.on_chain_error(e) for rm in rms])
            raise
        else:
            await asyncio.gather(
                *[rm.on_chain_end({"output": input}) for rm, input in zip(rms, inputs)]
            )
            return cast(List[Output], inputs)

    def stream(
        self, input: Input, config: Optional[RunnableConfig] = None
    ) -> Iterator[Output]:
        from langchain.callbacks.manager import CallbackManager

        # setup callbacks
        config = config or {}
        cm = CallbackManager.configure(
            inheritable_callbacks=config.get("callbacks"),
            local_callbacks=None,
            verbose=False,
            inheritable_tags=config.get("tags"),
            local_tags=None,
            inheritable_metadata=config.get("metadata"),
            local_metadata=None,
        )
        rm = cm.on_chain_start(dumpd(self), {"input": input})

        # invoke the first steps
        try:
            for step in [self.first] + self.middle:
                input = step.invoke(input, _patch_config(config, rm.get_child()))
        except (KeyboardInterrupt, Exception) as e:
            rm.on_chain_error(e)
            raise

        # stream the last step
        final: Union[Output, None] = None
        final_supported = True
        try:
            for output in self.last.stream(
                input, _patch_config(config, rm.get_child())
            ):
                yield output
                # Accumulate output if possible, otherwise disable accumulation
                if final_supported:
                    if final is None:
                        final = output
                    else:
                        try:
                            final += output  # type: ignore[operator]
                        except TypeError:
                            final = None
                            final_supported = False
                            pass
        except (KeyboardInterrupt, Exception) as e:
            rm.on_chain_error(e)
            raise
        else:
            rm.on_chain_end({"output": final})

    async def astream(
        self, input: Input, config: Optional[RunnableConfig] = None
    ) -> AsyncIterator[Output]:
        from langchain.callbacks.manager import AsyncCallbackManager

        # setup callbacks
        config = config or {}
        cm = AsyncCallbackManager.configure(
            inheritable_callbacks=config.get("callbacks"),
            local_callbacks=None,
            verbose=False,
            inheritable_tags=config.get("tags"),
            local_tags=None,
            inheritable_metadata=config.get("metadata"),
            local_metadata=None,
        )
        rm = await cm.on_chain_start(dumpd(self), {"input": input})

        # invoke the first steps
        try:
            for step in [self.first] + self.middle:
                input = await step.ainvoke(input, _patch_config(config, rm.get_child()))
        except (KeyboardInterrupt, Exception) as e:
            await rm.on_chain_error(e)
            raise

        # stream the last step
        final: Union[Output, None] = None
        final_supported = True
        try:
            async for output in self.last.astream(
                input, _patch_config(config, rm.get_child())
            ):
                yield output
                # Accumulate output if possible, otherwise disable accumulation
                if final_supported:
                    if final is None:
                        final = output
                    else:
                        try:
                            final += output  # type: ignore[operator]
                        except TypeError:
                            final = None
                            final_supported = False
                            pass
        except (KeyboardInterrupt, Exception) as e:
            await rm.on_chain_error(e)
            raise
        else:
            await rm.on_chain_end({"output": final})


def _patch_config(
    config: RunnableConfig, callback_manager: BaseCallbackManager
) -> RunnableConfig:
    config = config.copy()
    config["callbacks"] = callback_manager
    return config
