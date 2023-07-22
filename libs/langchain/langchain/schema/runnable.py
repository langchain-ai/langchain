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
)

from langchain.callbacks.manager import Callbacks


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


class Runnable(Generic[Input, Output], ABC):
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
        max_concurrency: Optional[int] = None,
    ) -> List[Output]:
        configs = self._get_config_list(config, len(inputs))
        coros = (self.ainvoke(input, config) for input, config in zip(inputs, configs))

        return await _gather_with_concurrency(max_concurrency, *coros)

    def stream(self, input: Input, config: RunnableConfig) -> Iterator[Output]:
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
