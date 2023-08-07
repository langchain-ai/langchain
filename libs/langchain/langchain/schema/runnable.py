from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Dict,
    Generic,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
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
    tags: List[str]
    """
    Tags for this call and any sub-calls (eg. a Chain calling an LLM).
    You can use these to filter calls.
    """

    metadata: Dict[str, Any]
    """
    Metadata for this call and any sub-calls (eg. a Chain calling an LLM).
    Keys should be strings, values should be JSON-serializable.
    """

    callbacks: Callbacks
    """
    Callbacks for this call and any sub-calls (eg. a Chain calling an LLM).
    Tags are passed to all callbacks, metadata is passed to handle*Start callbacks.
    """


Input = TypeVar("Input")
# Output type should implement __concat__, as eg str, list, dict do
Output = TypeVar("Output")
Other = TypeVar("Other")


class Runnable(Generic[Input, Output], ABC):
    def __or__(
        self,
        other: Union[
            Runnable[Any, Other],
            Callable[[Any], Other],
            Mapping[str, Union[Runnable[Any, Other], Callable[[Any], Other]]],
        ],
    ) -> RunnableSequence[Input, Other]:
        return RunnableSequence(first=self, last=_coerce_to_runnable(other))

    def __ror__(
        self,
        other: Union[
            Runnable[Other, Any],
            Callable[[Any], Other],
            Mapping[str, Union[Runnable[Other, Any], Callable[[Other], Any]]],
        ],
    ) -> RunnableSequence[Other, Output]:
        return RunnableSequence(first=_coerce_to_runnable(other), last=self)

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

        # If there's only one input, don't bother with the executor
        if len(inputs) == 1:
            return [self.invoke(inputs[0], configs[0])]

        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            return list(executor.map(self.invoke, inputs, configs))

    async def abatch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        max_concurrency: Optional[int] = None,
    ) -> List[Output]:
        configs = self._get_config_list(config, len(inputs))
        coros = map(self.ainvoke, inputs, configs)

        return await _gather_with_concurrency(max_concurrency, *coros)

    def stream(
        self, input: Input, config: Optional[RunnableConfig] = None
    ) -> Iterator[Output]:
        yield self.invoke(input, config)

    async def astream(
        self, input: Input, config: Optional[RunnableConfig] = None
    ) -> AsyncIterator[Output]:
        yield await self.ainvoke(input, config)

    def bind(self, **kwargs: Any) -> Runnable[Input, Output]:
        """
        Bind arguments to a Runnable, returning a new Runnable.
        """
        return RunnableBinding(bound=self, kwargs=kwargs)

    def _get_config_list(
        self, config: Optional[Union[RunnableConfig, List[RunnableConfig]]], length: int
    ) -> List[RunnableConfig]:
        if isinstance(config, list) and len(config) != length:
            raise ValueError(
                f"config must be a list of the same length as inputs, "
                f"but got {len(config)} configs for {length} inputs"
            )

        return (
            config
            if isinstance(config, list)
            else [config.copy() if config is not None else {} for _ in range(length)]
        )

    def _call_with_config(
        self,
        func: Callable[[Input], Output],
        input: Input,
        config: Optional[RunnableConfig],
        run_type: Optional[str] = None,
    ) -> Output:
        from langchain.callbacks.manager import CallbackManager

        config = config or {}
        callback_manager = CallbackManager.configure(
            inheritable_callbacks=config.get("callbacks"),
            inheritable_tags=config.get("tags"),
            inheritable_metadata=config.get("metadata"),
        )
        run_manager = callback_manager.on_chain_start(
            dumpd(self),
            input if isinstance(input, dict) else {"input": input},
            run_type=run_type,
        )
        try:
            output = func(input)
        except Exception as e:
            run_manager.on_chain_error(e)
            raise
        else:
            run_manager.on_chain_end(
                output if isinstance(output, dict) else {"output": output}
            )
            return output

    def with_fallbacks(
        self,
        fallbacks: Sequence[Runnable[Input, Output]],
        *,
        exceptions_to_handle: Tuple[Type[BaseException]] = (Exception,),
    ) -> RunnableWithFallbacks[Input, Output]:
        return RunnableWithFallbacks(
            runnable=self,
            fallbacks=fallbacks,
            exceptions_to_handle=exceptions_to_handle,
        )


class RunnableWithFallbacks(Serializable, Runnable[Input, Output]):
    runnable: Runnable[Input, Output]
    fallbacks: Sequence[Runnable[Input, Output]]
    exceptions_to_handle: Tuple[Type[BaseException]] = (Exception,)

    class Config:
        arbitrary_types_allowed = True

    @property
    def runnables(self) -> Iterator[Runnable[Input, Output]]:
        yield self.runnable
        yield from self.fallbacks

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        from langchain.callbacks.manager import CallbackManager

        # setup callbacks
        config = config or {}
        callback_manager = CallbackManager.configure(
            inheritable_callbacks=config.get("callbacks"),
            local_callbacks=None,
            verbose=False,
            inheritable_tags=config.get("tags"),
            local_tags=None,
            inheritable_metadata=config.get("metadata"),
            local_metadata=None,
        )
        # start the root run
        run_manager = callback_manager.on_chain_start(
            dumpd(self), input if isinstance(input, dict) else {"input": input}
        )
        first_error = None
        for runnable in self.runnables:
            try:
                output = runnable.invoke(
                    input,
                    _patch_config(config, run_manager.get_child()),
                )
            except self.exceptions_to_handle as e:
                if first_error is None:
                    first_error = e
            except BaseException as e:
                run_manager.on_chain_error(e)
                raise e
            else:
                run_manager.on_chain_end(
                    output if isinstance(output, dict) else {"output": output}
                )
                return output
        if first_error is None:
            raise ValueError("No error stored at end of fallbacks.")
        run_manager.on_chain_error(first_error)
        raise first_error

    async def ainvoke(
        self, input: Input, config: Optional[RunnableConfig] = None
    ) -> Output:
        from langchain.callbacks.manager import AsyncCallbackManager

        # setup callbacks
        config = config or {}
        callback_manager = AsyncCallbackManager.configure(
            inheritable_callbacks=config.get("callbacks"),
            local_callbacks=None,
            verbose=False,
            inheritable_tags=config.get("tags"),
            local_tags=None,
            inheritable_metadata=config.get("metadata"),
            local_metadata=None,
        )
        # start the root run
        run_manager = await callback_manager.on_chain_start(
            dumpd(self), input if isinstance(input, dict) else {"input": input}
        )

        first_error = None
        for runnable in self.runnables:
            try:
                output = await runnable.ainvoke(
                    input,
                    _patch_config(config, run_manager.get_child()),
                )
            except self.exceptions_to_handle as e:
                if first_error is None:
                    first_error = e
            except BaseException as e:
                await run_manager.on_chain_error(e)
                raise e
            else:
                await run_manager.on_chain_end(
                    output if isinstance(output, dict) else {"output": output}
                )
                return output
        if first_error is None:
            raise ValueError("No error stored at end of fallbacks.")
        await run_manager.on_chain_error(first_error)
        raise first_error

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
        callback_managers = [
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
        # start the root runs, one per input
        run_managers = [
            cm.on_chain_start(
                dumpd(self), input if isinstance(input, dict) else {"input": input}
            )
            for cm, input in zip(callback_managers, inputs)
        ]

        first_error = None
        for runnable in self.runnables:
            try:
                outputs = runnable.batch(
                    inputs,
                    [
                        # each step a child run of the corresponding root run
                        _patch_config(config, rm.get_child())
                        for rm, config in zip(run_managers, configs)
                    ],
                    max_concurrency=max_concurrency,
                )
            except self.exceptions_to_handle as e:
                if first_error is None:
                    first_error = e
            except BaseException as e:
                for rm in run_managers:
                    rm.on_chain_error(e)
                raise e
            else:
                for rm, output in zip(run_managers, outputs):
                    rm.on_chain_end(
                        output if isinstance(output, dict) else {"output": output}
                    )
                return outputs
        if first_error is None:
            raise ValueError("No error stored at end of fallbacks.")
        for rm in run_managers:
            rm.on_chain_error(first_error)
        raise first_error

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
        callback_managers = [
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
        # start the root runs, one per input
        run_managers: List[AsyncCallbackManagerForChainRun] = await asyncio.gather(
            *(
                cm.on_chain_start(
                    dumpd(self), input if isinstance(input, dict) else {"input": input}
                )
                for cm, input in zip(callback_managers, inputs)
            )
        )

        first_error = None
        for runnable in self.runnables:
            try:
                outputs = await runnable.abatch(
                    inputs,
                    [
                        # each step a child run of the corresponding root run
                        _patch_config(config, rm.get_child())
                        for rm, config in zip(run_managers, configs)
                    ],
                    max_concurrency=max_concurrency,
                )
            except self.exceptions_to_handle as e:
                if first_error is None:
                    first_error = e
            except BaseException as e:
                await asyncio.gather(*(rm.on_chain_error(e) for rm in run_managers))
            else:
                await asyncio.gather(
                    *(
                        rm.on_chain_end(
                            output if isinstance(output, dict) else {"output": output}
                        )
                        for rm, output in zip(run_managers, outputs)
                    )
                )
                return outputs
        if first_error is None:
            raise ValueError("No error stored at end of fallbacks.")
        await asyncio.gather(*(rm.on_chain_error(first_error) for rm in run_managers))
        raise first_error


class RunnableSequence(Serializable, Runnable[Input, Output]):
    first: Runnable[Input, Any]
    middle: List[Runnable[Any, Any]] = Field(default_factory=list)
    last: Runnable[Any, Output]

    @property
    def steps(self) -> List[Runnable[Any, Any]]:
        return [self.first] + self.middle + [self.last]

    @property
    def lc_serializable(self) -> bool:
        return True

    class Config:
        arbitrary_types_allowed = True

    def __or__(
        self,
        other: Union[
            Runnable[Any, Other],
            Callable[[Any], Other],
            Mapping[str, Union[Runnable[Any, Other], Callable[[Any], Other]]],
        ],
    ) -> RunnableSequence[Input, Other]:
        if isinstance(other, RunnableSequence):
            return RunnableSequence(
                first=self.first,
                middle=self.middle + [self.last] + [other.first] + other.middle,
                last=other.last,
            )
        else:
            return RunnableSequence(
                first=self.first,
                middle=self.middle + [self.last],
                last=_coerce_to_runnable(other),
            )

    def __ror__(
        self,
        other: Union[
            Runnable[Other, Any],
            Callable[[Any], Other],
            Mapping[str, Union[Runnable[Other, Any], Callable[[Other], Any]]],
        ],
    ) -> RunnableSequence[Other, Output]:
        if isinstance(other, RunnableSequence):
            return RunnableSequence(
                first=other.first,
                middle=other.middle + [other.last] + [self.first] + self.middle,
                last=self.last,
            )
        else:
            return RunnableSequence(
                first=_coerce_to_runnable(other),
                middle=[self.first] + self.middle,
                last=self.last,
            )

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        from langchain.callbacks.manager import CallbackManager

        # setup callbacks
        config = config or {}
        callback_manager = CallbackManager.configure(
            inheritable_callbacks=config.get("callbacks"),
            local_callbacks=None,
            verbose=False,
            inheritable_tags=config.get("tags"),
            local_tags=None,
            inheritable_metadata=config.get("metadata"),
            local_metadata=None,
        )
        # start the root run
        run_manager = callback_manager.on_chain_start(
            dumpd(self), input if isinstance(input, dict) else {"input": input}
        )

        # invoke all steps in sequence
        try:
            for step in self.steps:
                input = step.invoke(
                    input,
                    # mark each step as a child run
                    _patch_config(config, run_manager.get_child()),
                )
        # finish the root run
        except (KeyboardInterrupt, Exception) as e:
            run_manager.on_chain_error(e)
            raise
        else:
            run_manager.on_chain_end(
                input if isinstance(input, dict) else {"output": input}
            )
            return cast(Output, input)

    async def ainvoke(
        self, input: Input, config: Optional[RunnableConfig] = None
    ) -> Output:
        from langchain.callbacks.manager import AsyncCallbackManager

        # setup callbacks
        config = config or {}
        callback_manager = AsyncCallbackManager.configure(
            inheritable_callbacks=config.get("callbacks"),
            local_callbacks=None,
            verbose=False,
            inheritable_tags=config.get("tags"),
            local_tags=None,
            inheritable_metadata=config.get("metadata"),
            local_metadata=None,
        )
        # start the root run
        run_manager = await callback_manager.on_chain_start(
            dumpd(self), input if isinstance(input, dict) else {"input": input}
        )

        # invoke all steps in sequence
        try:
            for step in self.steps:
                input = await step.ainvoke(
                    input,
                    # mark each step as a child run
                    _patch_config(config, run_manager.get_child()),
                )
        # finish the root run
        except (KeyboardInterrupt, Exception) as e:
            await run_manager.on_chain_error(e)
            raise
        else:
            await run_manager.on_chain_end(
                input if isinstance(input, dict) else {"output": input}
            )
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
        callback_managers = [
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
        # start the root runs, one per input
        run_managers = [
            cm.on_chain_start(
                dumpd(self), input if isinstance(input, dict) else {"input": input}
            )
            for cm, input in zip(callback_managers, inputs)
        ]

        # invoke
        try:
            for step in self.steps:
                inputs = step.batch(
                    inputs,
                    [
                        # each step a child run of the corresponding root run
                        _patch_config(config, rm.get_child())
                        for rm, config in zip(run_managers, configs)
                    ],
                    max_concurrency=max_concurrency,
                )
        # finish the root runs
        except (KeyboardInterrupt, Exception) as e:
            for rm in run_managers:
                rm.on_chain_error(e)
            raise
        else:
            for rm, input in zip(run_managers, inputs):
                rm.on_chain_end(input if isinstance(input, dict) else {"output": input})
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
        callback_managers = [
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
        # start the root runs, one per input
        run_managers: List[AsyncCallbackManagerForChainRun] = await asyncio.gather(
            *(
                cm.on_chain_start(
                    dumpd(self), input if isinstance(input, dict) else {"input": input}
                )
                for cm, input in zip(callback_managers, inputs)
            )
        )

        # invoke .batch() on each step
        # this uses batching optimizations in Runnable subclasses, like LLM
        try:
            for step in self.steps:
                inputs = await step.abatch(
                    inputs,
                    [
                        # each step a child run of the corresponding root run
                        _patch_config(config, rm.get_child())
                        for rm, config in zip(run_managers, configs)
                    ],
                    max_concurrency=max_concurrency,
                )
        # finish the root runs
        except (KeyboardInterrupt, Exception) as e:
            await asyncio.gather(*(rm.on_chain_error(e) for rm in run_managers))
            raise
        else:
            await asyncio.gather(
                *(
                    rm.on_chain_end(
                        input if isinstance(input, dict) else {"output": input}
                    )
                    for rm, input in zip(run_managers, inputs)
                )
            )
            return cast(List[Output], inputs)

    def stream(
        self, input: Input, config: Optional[RunnableConfig] = None
    ) -> Iterator[Output]:
        from langchain.callbacks.manager import CallbackManager

        # setup callbacks
        config = config or {}
        callback_manager = CallbackManager.configure(
            inheritable_callbacks=config.get("callbacks"),
            local_callbacks=None,
            verbose=False,
            inheritable_tags=config.get("tags"),
            local_tags=None,
            inheritable_metadata=config.get("metadata"),
            local_metadata=None,
        )
        # start the root run
        run_manager = callback_manager.on_chain_start(
            dumpd(self), input if isinstance(input, dict) else {"input": input}
        )

        # invoke the first steps
        try:
            for step in [self.first] + self.middle:
                input = step.invoke(
                    input,
                    # mark each step as a child run
                    _patch_config(config, run_manager.get_child()),
                )
        except (KeyboardInterrupt, Exception) as e:
            run_manager.on_chain_error(e)
            raise

        # stream the last step
        final: Union[Output, None] = None
        final_supported = True
        try:
            for output in self.last.stream(
                input,
                # mark the last step as a child run
                _patch_config(config, run_manager.get_child()),
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
        # finish the root run
        except (KeyboardInterrupt, Exception) as e:
            run_manager.on_chain_error(e)
            raise
        else:
            run_manager.on_chain_end(
                final if isinstance(final, dict) else {"output": final}
            )

    async def astream(
        self, input: Input, config: Optional[RunnableConfig] = None
    ) -> AsyncIterator[Output]:
        from langchain.callbacks.manager import AsyncCallbackManager

        # setup callbacks
        config = config or {}
        callback_manager = AsyncCallbackManager.configure(
            inheritable_callbacks=config.get("callbacks"),
            local_callbacks=None,
            verbose=False,
            inheritable_tags=config.get("tags"),
            local_tags=None,
            inheritable_metadata=config.get("metadata"),
            local_metadata=None,
        )
        # start the root run
        run_manager = await callback_manager.on_chain_start(
            dumpd(self), input if isinstance(input, dict) else {"input": input}
        )

        # invoke the first steps
        try:
            for step in [self.first] + self.middle:
                input = await step.ainvoke(
                    input,
                    # mark each step as a child run
                    _patch_config(config, run_manager.get_child()),
                )
        except (KeyboardInterrupt, Exception) as e:
            await run_manager.on_chain_error(e)
            raise

        # stream the last step
        final: Union[Output, None] = None
        final_supported = True
        try:
            async for output in self.last.astream(
                input,
                # mark the last step as a child run
                _patch_config(config, run_manager.get_child()),
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
        # finish the root run
        except (KeyboardInterrupt, Exception) as e:
            await run_manager.on_chain_error(e)
            raise
        else:
            await run_manager.on_chain_end(
                final if isinstance(final, dict) else {"output": final}
            )


class RunnableMap(Serializable, Runnable[Input, Dict[str, Any]]):
    steps: Mapping[str, Runnable[Input, Any]]

    def __init__(
        self,
        steps: Mapping[
            str,
            Union[
                Runnable[Input, Any],
                Callable[[Input], Any],
                Mapping[str, Union[Runnable[Input, Any], Callable[[Input], Any]]],
            ],
        ],
    ) -> None:
        super().__init__(
            steps={key: _coerce_to_runnable(r) for key, r in steps.items()}
        )

    @property
    def lc_serializable(self) -> bool:
        return True

    class Config:
        arbitrary_types_allowed = True

    def invoke(
        self, input: Input, config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        from langchain.callbacks.manager import CallbackManager

        # setup callbacks
        config = config or {}
        callback_manager = CallbackManager.configure(
            inheritable_callbacks=config.get("callbacks"),
            local_callbacks=None,
            verbose=False,
            inheritable_tags=config.get("tags"),
            local_tags=None,
            inheritable_metadata=config.get("metadata"),
            local_metadata=None,
        )
        # start the root run
        run_manager = callback_manager.on_chain_start(dumpd(self), {"input": input})

        # gather results from all steps
        try:
            # copy to avoid issues from the caller mutating the steps during invoke()
            steps = dict(self.steps)
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        step.invoke,
                        input,
                        # mark each step as a child run
                        _patch_config(config, run_manager.get_child()),
                    )
                    for step in steps.values()
                ]
                output = {key: future.result() for key, future in zip(steps, futures)}
        # finish the root run
        except (KeyboardInterrupt, Exception) as e:
            run_manager.on_chain_error(e)
            raise
        else:
            run_manager.on_chain_end(output)
            return output

    async def ainvoke(
        self, input: Input, config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        from langchain.callbacks.manager import AsyncCallbackManager

        # setup callbacks
        config = config or {}
        callback_manager = AsyncCallbackManager.configure(
            inheritable_callbacks=config.get("callbacks"),
            local_callbacks=None,
            verbose=False,
            inheritable_tags=config.get("tags"),
            local_tags=None,
            inheritable_metadata=config.get("metadata"),
            local_metadata=None,
        )
        # start the root run
        run_manager = await callback_manager.on_chain_start(
            dumpd(self), {"input": input}
        )

        # gather results from all steps
        try:
            # copy to avoid issues from the caller mutating the steps during invoke()
            steps = dict(self.steps)
            results = await asyncio.gather(
                *(
                    step.ainvoke(
                        input,
                        # mark each step as a child run
                        _patch_config(config, run_manager.get_child()),
                    )
                    for step in steps.values()
                )
            )
            output = {key: value for key, value in zip(steps, results)}
        # finish the root run
        except (KeyboardInterrupt, Exception) as e:
            await run_manager.on_chain_error(e)
            raise
        else:
            await run_manager.on_chain_end(output)
            return output


class RunnableLambda(Runnable[Input, Output]):
    def __init__(self, func: Callable[[Input], Output]) -> None:
        if callable(func):
            self.func = func
        else:
            raise TypeError(
                "Expected a callable type for `func`."
                f"Instead got an unsupported type: {type(func)}"
            )

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, RunnableLambda):
            return self.func == other.func
        else:
            return False

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        return self._call_with_config(self.func, input, config)


class RunnablePassthrough(Serializable, Runnable[Input, Input]):
    @property
    def lc_serializable(self) -> bool:
        return True

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Input:
        return self._call_with_config(lambda x: x, input, config)


class RunnableBinding(Serializable, Runnable[Input, Output]):
    bound: Runnable[Input, Output]

    kwargs: Mapping[str, Any]

    class Config:
        arbitrary_types_allowed = True

    @property
    def lc_serializable(self) -> bool:
        return True

    def bind(self, **kwargs: Any) -> Runnable[Input, Output]:
        return self.__class__(bound=self.bound, kwargs={**self.kwargs, **kwargs})

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        return self.bound.invoke(input, config, **self.kwargs)

    async def ainvoke(
        self, input: Input, config: Optional[RunnableConfig] = None
    ) -> Output:
        return await self.bound.ainvoke(input, config, **self.kwargs)

    def batch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        max_concurrency: Optional[int] = None,
    ) -> List[Output]:
        return self.bound.batch(
            inputs, config, max_concurrency=max_concurrency, **self.kwargs
        )

    async def abatch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        max_concurrency: Optional[int] = None,
    ) -> List[Output]:
        return await self.bound.abatch(
            inputs, config, max_concurrency=max_concurrency, **self.kwargs
        )

    def stream(
        self, input: Input, config: Optional[RunnableConfig] = None
    ) -> Iterator[Output]:
        yield from self.bound.stream(input, config, **self.kwargs)

    async def astream(
        self, input: Input, config: Optional[RunnableConfig] = None
    ) -> AsyncIterator[Output]:
        async for item in self.bound.astream(input, config, **self.kwargs):
            yield item


class RouterInput(TypedDict):
    key: str
    input: Any


class RouterRunnable(
    Serializable, Generic[Input, Output], Runnable[RouterInput, Output]
):
    runnables: Mapping[str, Runnable[Input, Output]]

    def __init__(self, runnables: Mapping[str, Runnable[Input, Output]]) -> None:
        super().__init__(runnables=runnables)

    class Config:
        arbitrary_types_allowed = True

    @property
    def lc_serializable(self) -> bool:
        return True

    def __or__(
        self,
        other: Union[
            Runnable[Any, Other],
            Callable[[Any], Other],
            Mapping[str, Union[Runnable[Any, Other], Callable[[Any], Other]]],
            Mapping[str, Any],
        ],
    ) -> RunnableSequence[RouterInput, Other]:
        return RunnableSequence(first=self, last=_coerce_to_runnable(other))

    def __ror__(
        self,
        other: Union[
            Runnable[Other, Any],
            Callable[[Any], Other],
            Mapping[str, Union[Runnable[Other, Any], Callable[[Other], Any]]],
            Mapping[str, Any],
        ],
    ) -> RunnableSequence[Other, Output]:
        return RunnableSequence(first=_coerce_to_runnable(other), last=self)

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
        self, input: RouterInput, config: Optional[RunnableConfig] = None
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
        max_concurrency: Optional[int] = None,
    ) -> List[Output]:
        keys = [input["key"] for input in inputs]
        actual_inputs = [input["input"] for input in inputs]
        if any(key not in self.runnables for key in keys):
            raise ValueError("One or more keys do not have a corresponding runnable")

        runnables = [self.runnables[key] for key in keys]
        configs = self._get_config_list(config, len(inputs))
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            return list(
                executor.map(
                    lambda runnable, input, config: runnable.invoke(input, config),
                    runnables,
                    actual_inputs,
                    configs,
                )
            )

    async def abatch(
        self,
        inputs: List[RouterInput],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        max_concurrency: Optional[int] = None,
    ) -> List[Output]:
        keys = [input["key"] for input in inputs]
        actual_inputs = [input["input"] for input in inputs]
        if any(key not in self.runnables for key in keys):
            raise ValueError("One or more keys do not have a corresponding runnable")

        runnables = [self.runnables[key] for key in keys]
        configs = self._get_config_list(config, len(inputs))
        return await _gather_with_concurrency(
            max_concurrency,
            *(
                runnable.ainvoke(input, config)
                for runnable, input, config in zip(runnables, actual_inputs, configs)
            ),
        )

    def stream(
        self, input: RouterInput, config: Optional[RunnableConfig] = None
    ) -> Iterator[Output]:
        key = input["key"]
        actual_input = input["input"]
        if key not in self.runnables:
            raise ValueError(f"No runnable associated with key '{key}'")

        runnable = self.runnables[key]
        yield from runnable.stream(actual_input, config)

    async def astream(
        self, input: RouterInput, config: Optional[RunnableConfig] = None
    ) -> AsyncIterator[Output]:
        key = input["key"]
        actual_input = input["input"]
        if key not in self.runnables:
            raise ValueError(f"No runnable associated with key '{key}'")

        runnable = self.runnables[key]
        async for output in runnable.astream(actual_input, config):
            yield output


def _patch_config(
    config: RunnableConfig, callback_manager: BaseCallbackManager
) -> RunnableConfig:
    config = config.copy()
    config["callbacks"] = callback_manager
    return config


def _coerce_to_runnable(
    thing: Union[
        Runnable[Input, Output],
        Callable[[Input], Output],
        Mapping[str, Union[Runnable[Input, Output], Callable[[Input], Output]]],
    ]
) -> Runnable[Input, Output]:
    if isinstance(thing, Runnable):
        return thing
    elif callable(thing):
        return RunnableLambda(thing)
    elif isinstance(thing, dict):
        runnables = {key: _coerce_to_runnable(r) for key, r in thing.items()}
        return cast(Runnable[Input, Output], RunnableMap(steps=runnables))
    else:
        raise TypeError(
            f"Expected a Runnable, callable or dict."
            f"Instead got an unsupported type: {type(thing)}"
        )
