from __future__ import annotations

import asyncio
import copy
import threading
from abc import ABC, abstractmethod
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from functools import partial
from itertools import tee
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

if TYPE_CHECKING:
    from langchain.callbacks.manager import (
        AsyncCallbackManagerForChainRun,
        CallbackManagerForChainRun,
    )


from langchain.callbacks.base import BaseCallbackManager
from langchain.load.dump import dumpd
from langchain.load.serializable import Serializable
from langchain.pydantic_v1 import Field
from langchain.schema.runnable.config import RunnableConfig
from langchain.schema.runnable.utils import (
    accepts_run_manager,
    accepts_run_manager_and_config,
    gather_with_concurrency,
)
from langchain.utils.aiter import atee, py_anext
from langchain.utils.iter import safetee

Input = TypeVar("Input")
# Output type should implement __concat__, as eg str, list, dict do
Output = TypeVar("Output")
Other = TypeVar("Other")


class Runnable(Generic[Input, Output], ABC):
    """A Runnable is a unit of work that can be invoked, batched, streamed, or
    transformed."""

    def __or__(
        self,
        other: Union[
            Runnable[Any, Other],
            Callable[[Any], Other],
            Mapping[str, Union[Runnable[Any, Other], Callable[[Any], Other], Any]],
        ],
    ) -> RunnableSequence[Input, Other]:
        return RunnableSequence(first=self, last=coerce_to_runnable(other))

    def __ror__(
        self,
        other: Union[
            Runnable[Other, Any],
            Callable[[Any], Other],
            Mapping[str, Union[Runnable[Other, Any], Callable[[Other], Any], Any]],
        ],
    ) -> RunnableSequence[Other, Output]:
        return RunnableSequence(first=coerce_to_runnable(other), last=self)

    """ --- Public API --- """

    @abstractmethod
    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        ...

    async def ainvoke(
        self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Output:
        """
        Default implementation of ainvoke, which calls invoke in a thread pool.
        Subclasses should override this method if they can run asynchronously.
        """
        return await asyncio.get_running_loop().run_in_executor(
            None, partial(self.invoke, **kwargs), input, config
        )

    def batch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        max_concurrency: Optional[int] = None,
        **kwargs: Optional[Any],
    ) -> List[Output]:
        """
        Default implementation of batch, which calls invoke N times.
        Subclasses should override this method if they can batch more efficiently.
        """
        configs = self._get_config_list(config, len(inputs))

        # If there's only one input, don't bother with the executor
        if len(inputs) == 1:
            return [self.invoke(inputs[0], configs[0], **kwargs)]

        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            return list(executor.map(partial(self.invoke, **kwargs), inputs, configs))

    async def abatch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        max_concurrency: Optional[int] = None,
        **kwargs: Optional[Any],
    ) -> List[Output]:
        """
        Default implementation of abatch, which calls ainvoke N times.
        Subclasses should override this method if they can batch more efficiently.
        """
        configs = self._get_config_list(config, len(inputs))
        coros = map(partial(self.ainvoke, **kwargs), inputs, configs)

        return await gather_with_concurrency(max_concurrency, *coros)

    def stream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Output]:
        """
        Default implementation of stream, which calls invoke.
        Subclasses should override this method if they support streaming output.
        """
        yield self.invoke(input, config, **kwargs)

    async def astream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Output]:
        """
        Default implementation of astream, which calls ainvoke.
        Subclasses should override this method if they support streaming output.
        """
        yield await self.ainvoke(input, config, **kwargs)

    def transform(
        self,
        input: Iterator[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Output]:
        """
        Default implementation of transform, which buffers input and then calls stream.
        Subclasses should override this method if they can start producing output while
        input is still being generated.
        """
        final: Union[Input, None] = None

        for chunk in input:
            if final is None:
                final = chunk
            else:
                # Make a best effort to gather, for any type that supports `+`
                # This method should throw an error if gathering fails.
                final += chunk  # type: ignore[operator]
        if final:
            yield from self.stream(final, config, **kwargs)

    async def atransform(
        self,
        input: AsyncIterator[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Output]:
        """
        Default implementation of atransform, which buffers input and calls astream.
        Subclasses should override this method if they can start producing output while
        input is still being generated.
        """
        final: Union[Input, None] = None

        async for chunk in input:
            if final is None:
                final = chunk
            else:
                # Make a best effort to gather, for any type that supports `+`
                # This method should throw an error if gathering fails.
                final += chunk  # type: ignore[operator]

        if final:
            async for output in self.astream(final, config, **kwargs):
                yield output

    def bind(self, **kwargs: Any) -> Runnable[Input, Output]:
        """
        Bind arguments to a Runnable, returning a new Runnable.
        """
        return RunnableBinding(bound=self, kwargs=kwargs)

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

    """ --- Helper methods for Subclasses --- """

    def _get_config_list(
        self, config: Optional[Union[RunnableConfig, List[RunnableConfig]]], length: int
    ) -> List[RunnableConfig]:
        """
        Helper method to get a list of configs from a single config or a list of
        configs, useful for subclasses overriding batch() or abatch().
        """
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
        func: Union[
            Callable[[Input], Output],
            Callable[[Input, CallbackManagerForChainRun], Output],
            Callable[[Input, CallbackManagerForChainRun, RunnableConfig], Output],
        ],
        input: Input,
        config: Optional[RunnableConfig],
        run_type: Optional[str] = None,
    ) -> Output:
        """Helper method to transform an Input value to an Output value,
        with callbacks. Use this method to implement invoke() in subclasses."""
        from langchain.callbacks.manager import CallbackManager

        config = config or {}
        callback_manager = CallbackManager.configure(
            inheritable_callbacks=config.get("callbacks"),
            inheritable_tags=config.get("tags"),
            inheritable_metadata=config.get("metadata"),
        )
        run_manager = callback_manager.on_chain_start(
            dumpd(self),
            input,
            run_type=run_type,
        )
        try:
            if accepts_run_manager_and_config(func):
                output = func(
                    input,
                    run_manager=run_manager,
                    config=config,
                )  # type: ignore[call-arg]
            elif accepts_run_manager(func):
                output = func(input, run_manager=run_manager)  # type: ignore[call-arg]
            else:
                output = func(input)  # type: ignore[call-arg]
        except Exception as e:
            run_manager.on_chain_error(e)
            raise
        else:
            run_manager.on_chain_end(dumpd(output))
            return output

    async def _acall_with_config(
        self,
        func: Union[
            Callable[[Input], Awaitable[Output]],
            Callable[[Input, AsyncCallbackManagerForChainRun], Awaitable[Output]],
            Callable[
                [Input, AsyncCallbackManagerForChainRun, RunnableConfig],
                Awaitable[Output],
            ],
        ],
        input: Input,
        config: Optional[RunnableConfig],
        run_type: Optional[str] = None,
    ) -> Output:
        """Helper method to transform an Input value to an Output value,
        with callbacks. Use this method to implement ainvoke() in subclasses."""
        from langchain.callbacks.manager import AsyncCallbackManager

        config = config or {}
        callback_manager = AsyncCallbackManager.configure(
            inheritable_callbacks=config.get("callbacks"),
            inheritable_tags=config.get("tags"),
            inheritable_metadata=config.get("metadata"),
        )
        run_manager = await callback_manager.on_chain_start(
            dumpd(self),
            input,
            run_type=run_type,
        )
        try:
            if accepts_run_manager_and_config(func):
                output = await func(
                    input,
                    run_manager=run_manager,
                    config=config,
                )  # type: ignore[call-arg]
            elif accepts_run_manager(func):
                output = await func(
                    input,
                    run_manager=run_manager,
                )  # type: ignore[call-arg]
            else:
                output = await func(input)  # type: ignore[call-arg]
        except Exception as e:
            await run_manager.on_chain_error(e)
            raise
        else:
            await run_manager.on_chain_end(dumpd(output))
            return output

    def _transform_stream_with_config(
        self,
        input: Iterator[Input],
        transformer: Union[
            Callable[[Iterator[Input]], Iterator[Output]],
            Callable[[Iterator[Input], CallbackManagerForChainRun], Iterator[Output]],
            Callable[
                [
                    Iterator[Input],
                    CallbackManagerForChainRun,
                    RunnableConfig,
                ],
                Iterator[Output],
            ],
        ],
        config: Optional[RunnableConfig],
        run_type: Optional[str] = None,
    ) -> Iterator[Output]:
        """Helper method to transform an Iterator of Input values into an Iterator of
        Output values, with callbacks.
        Use this to implement `stream()` or `transform()` in Runnable subclasses."""
        from langchain.callbacks.manager import CallbackManager

        # tee the input so we can iterate over it twice
        input_for_tracing, input_for_transform = tee(input, 2)
        # Start the input iterator to ensure the input runnable starts before this one
        final_input: Optional[Input] = next(input_for_tracing, None)
        final_input_supported = True
        final_output: Optional[Output] = None
        final_output_supported = True

        config = config or {}
        callback_manager = CallbackManager.configure(
            inheritable_callbacks=config.get("callbacks"),
            inheritable_tags=config.get("tags"),
            inheritable_metadata=config.get("metadata"),
        )
        run_manager = callback_manager.on_chain_start(
            dumpd(self),
            {"input": ""},
            run_type=run_type,
        )
        try:
            if accepts_run_manager_and_config(transformer):
                iterator = transformer(
                    input_for_transform,
                    run_manager=run_manager,
                    config=config,
                )  # type: ignore[call-arg]
            elif accepts_run_manager(transformer):
                iterator = transformer(
                    input_for_transform,
                    run_manager=run_manager,
                )  # type: ignore[call-arg]
            else:
                iterator = transformer(input_for_transform)  # type: ignore[call-arg]
            for chunk in iterator:
                yield chunk
                if final_output_supported:
                    if final_output is None:
                        final_output = chunk
                    else:
                        try:
                            final_output += chunk  # type: ignore[operator]
                        except TypeError:
                            final_output = None
                            final_output_supported = False
            for ichunk in input_for_tracing:
                if final_input_supported:
                    if final_input is None:
                        final_input = ichunk
                    else:
                        try:
                            final_input += ichunk  # type: ignore[operator]
                        except TypeError:
                            final_input = None
                            final_input_supported = False
        except Exception as e:
            run_manager.on_chain_error(e, inputs=final_input)
            raise
        else:
            run_manager.on_chain_end(final_output, inputs=final_input)

    async def _atransform_stream_with_config(
        self,
        input: AsyncIterator[Input],
        transformer: Union[
            Callable[[AsyncIterator[Input]], AsyncIterator[Output]],
            Callable[
                [AsyncIterator[Input], AsyncCallbackManagerForChainRun],
                AsyncIterator[Output],
            ],
            Callable[
                [
                    AsyncIterator[Input],
                    AsyncCallbackManagerForChainRun,
                    RunnableConfig,
                ],
                AsyncIterator[Output],
            ],
        ],
        config: Optional[RunnableConfig],
        run_type: Optional[str] = None,
    ) -> AsyncIterator[Output]:
        """Helper method to transform an Async Iterator of Input values into an Async
        Iterator of Output values, with callbacks.
        Use this to implement `astream()` or `atransform()` in Runnable subclasses."""
        from langchain.callbacks.manager import AsyncCallbackManager

        # tee the input so we can iterate over it twice
        input_for_tracing, input_for_transform = atee(input, 2)
        # Start the input iterator to ensure the input runnable starts before this one
        final_input: Optional[Input] = await py_anext(input_for_tracing, None)
        final_input_supported = True
        final_output: Optional[Output] = None
        final_output_supported = True

        config = config or {}
        callback_manager = AsyncCallbackManager.configure(
            inheritable_callbacks=config.get("callbacks"),
            inheritable_tags=config.get("tags"),
            inheritable_metadata=config.get("metadata"),
        )
        run_manager = await callback_manager.on_chain_start(
            dumpd(self),
            {"input": ""},
            run_type=run_type,
        )
        try:
            # mypy can't quite work out thew type guard here, but this is safe,
            # check implementations of the accepts_* functions
            if accepts_run_manager_and_config(transformer):
                iterator = transformer(
                    input_for_transform,
                    run_manager=run_manager,
                    config=config,
                )  # type: ignore[call-arg]
            elif accepts_run_manager(transformer):
                iterator = transformer(
                    input_for_transform,
                    run_manager=run_manager,
                )  # type: ignore[call-arg]
            else:
                iterator = transformer(input_for_transform)  # type: ignore[call-arg]
            async for chunk in iterator:
                yield chunk
                if final_output_supported:
                    if final_output is None:
                        final_output = chunk
                    else:
                        try:
                            final_output += chunk  # type: ignore[operator]
                        except TypeError:
                            final_output = None
                            final_output_supported = False
            async for ichunk in input_for_tracing:
                if final_input_supported:
                    if final_input is None:
                        final_input = ichunk
                    else:
                        try:
                            final_input += ichunk  # type: ignore[operator]
                        except TypeError:
                            final_input = None
                            final_input_supported = False
        except Exception as e:
            await run_manager.on_chain_error(e, inputs=final_input)
            raise
        else:
            await run_manager.on_chain_end(final_output, inputs=final_input)


class RunnableWithFallbacks(Serializable, Runnable[Input, Output]):
    """
    A Runnable that can fallback to other Runnables if it fails.
    """

    runnable: Runnable[Input, Output]
    fallbacks: Sequence[Runnable[Input, Output]]
    exceptions_to_handle: Tuple[Type[BaseException]] = (Exception,)

    class Config:
        arbitrary_types_allowed = True

    @property
    def lc_serializable(self) -> bool:
        return True

    @property
    def lc_namespace(self) -> List[str]:
        return self.__class__.__module__.split(".")[:-1]

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
        run_manager = callback_manager.on_chain_start(dumpd(self), input)
        first_error = None
        for runnable in self.runnables:
            try:
                output = runnable.invoke(
                    input,
                    patch_config(config, run_manager.get_child()),
                )
            except self.exceptions_to_handle as e:
                if first_error is None:
                    first_error = e
            except BaseException as e:
                run_manager.on_chain_error(e)
                raise e
            else:
                run_manager.on_chain_end(output)
                return output
        if first_error is None:
            raise ValueError("No error stored at end of fallbacks.")
        run_manager.on_chain_error(first_error)
        raise first_error

    async def ainvoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
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
        run_manager = await callback_manager.on_chain_start(dumpd(self), input)

        first_error = None
        for runnable in self.runnables:
            try:
                output = await runnable.ainvoke(
                    input,
                    patch_config(config, run_manager.get_child()),
                )
            except self.exceptions_to_handle as e:
                if first_error is None:
                    first_error = e
            except BaseException as e:
                await run_manager.on_chain_error(e)
                raise e
            else:
                await run_manager.on_chain_end(output)
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
        **kwargs: Optional[Any],
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
                        patch_config(config, rm.get_child())
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
                    rm.on_chain_end(output)
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
        **kwargs: Optional[Any],
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
                cm.on_chain_start(dumpd(self), input)
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
                        patch_config(config, rm.get_child())
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
                        rm.on_chain_end(output)
                        for rm, output in zip(run_managers, outputs)
                    )
                )
                return outputs
        if first_error is None:
            raise ValueError("No error stored at end of fallbacks.")
        await asyncio.gather(*(rm.on_chain_error(first_error) for rm in run_managers))
        raise first_error


class RunnableSequence(Serializable, Runnable[Input, Output]):
    """
    A sequence of runnables, where the output of each is the input of the next.
    """

    first: Runnable[Input, Any]
    middle: List[Runnable[Any, Any]] = Field(default_factory=list)
    last: Runnable[Any, Output]

    @property
    def steps(self) -> List[Runnable[Any, Any]]:
        return [self.first] + self.middle + [self.last]

    @property
    def lc_serializable(self) -> bool:
        return True

    @property
    def lc_namespace(self) -> List[str]:
        return self.__class__.__module__.split(".")[:-1]

    class Config:
        arbitrary_types_allowed = True

    def __or__(
        self,
        other: Union[
            Runnable[Any, Other],
            Callable[[Any], Other],
            Mapping[str, Union[Runnable[Any, Other], Callable[[Any], Other], Any]],
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
                last=coerce_to_runnable(other),
            )

    def __ror__(
        self,
        other: Union[
            Runnable[Other, Any],
            Callable[[Any], Other],
            Mapping[str, Union[Runnable[Other, Any], Callable[[Other], Any], Any]],
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
                first=coerce_to_runnable(other),
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
        run_manager = callback_manager.on_chain_start(dumpd(self), input)

        # invoke all steps in sequence
        try:
            for step in self.steps:
                input = step.invoke(
                    input,
                    # mark each step as a child run
                    patch_config(config, run_manager.get_child()),
                )
        # finish the root run
        except (KeyboardInterrupt, Exception) as e:
            run_manager.on_chain_error(e)
            raise
        else:
            run_manager.on_chain_end(input)
            return cast(Output, input)

    async def ainvoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
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
        run_manager = await callback_manager.on_chain_start(dumpd(self), input)

        # invoke all steps in sequence
        try:
            for step in self.steps:
                input = await step.ainvoke(
                    input,
                    # mark each step as a child run
                    patch_config(config, run_manager.get_child()),
                )
        # finish the root run
        except (KeyboardInterrupt, Exception) as e:
            await run_manager.on_chain_error(e)
            raise
        else:
            await run_manager.on_chain_end(input)
            return cast(Output, input)

    def batch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        max_concurrency: Optional[int] = None,
        **kwargs: Optional[Any],
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
            cm.on_chain_start(dumpd(self), input)
            for cm, input in zip(callback_managers, inputs)
        ]

        # invoke
        try:
            for step in self.steps:
                inputs = step.batch(
                    inputs,
                    [
                        # each step a child run of the corresponding root run
                        patch_config(config, rm.get_child())
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
                rm.on_chain_end(input)
            return cast(List[Output], inputs)

    async def abatch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        max_concurrency: Optional[int] = None,
        **kwargs: Optional[Any],
    ) -> List[Output]:
        from langchain.callbacks.manager import (
            AsyncCallbackManager,
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
                cm.on_chain_start(dumpd(self), input)
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
                        patch_config(config, rm.get_child())
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
                *(rm.on_chain_end(input) for rm, input in zip(run_managers, inputs))
            )
            return cast(List[Output], inputs)

    def stream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
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
        run_manager = callback_manager.on_chain_start(dumpd(self), input)

        steps = [self.first] + self.middle + [self.last]
        streaming_start_index = 0

        for i in range(len(steps) - 1, 0, -1):
            if type(steps[i]).transform != Runnable.transform:
                streaming_start_index = i - 1
            else:
                break

        # invoke the first steps
        try:
            for step in steps[0:streaming_start_index]:
                input = step.invoke(
                    input,
                    # mark each step as a child run
                    patch_config(config, run_manager.get_child()),
                )
        except (KeyboardInterrupt, Exception) as e:
            run_manager.on_chain_error(e)
            raise

        # stream the last steps
        final: Union[Output, None] = None
        final_supported = True
        try:
            # stream the first of the last steps with non-streaming input
            final_pipeline = steps[streaming_start_index].stream(
                input, patch_config(config, run_manager.get_child())
            )
            # stream the rest of the last steps with streaming input
            for step in steps[streaming_start_index + 1 :]:
                final_pipeline = step.transform(
                    final_pipeline, patch_config(config, run_manager.get_child())
                )
            for output in final_pipeline:
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
            run_manager.on_chain_end(final)

    async def astream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
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
        run_manager = await callback_manager.on_chain_start(dumpd(self), input)

        steps = [self.first] + self.middle + [self.last]
        streaming_start_index = len(steps) - 1

        for i in range(len(steps) - 1, 0, -1):
            if type(steps[i]).transform != Runnable.transform:
                streaming_start_index = i - 1
            else:
                break

        # invoke the first steps
        try:
            for step in steps[0:streaming_start_index]:
                input = await step.ainvoke(
                    input,
                    # mark each step as a child run
                    patch_config(config, run_manager.get_child()),
                )
        except (KeyboardInterrupt, Exception) as e:
            await run_manager.on_chain_error(e)
            raise

        # stream the last steps
        final: Union[Output, None] = None
        final_supported = True
        try:
            # stream the first of the last steps with non-streaming input
            final_pipeline = steps[streaming_start_index].astream(
                input, patch_config(config, run_manager.get_child())
            )
            # stream the rest of the last steps with streaming input
            for step in steps[streaming_start_index + 1 :]:
                final_pipeline = step.atransform(
                    final_pipeline, patch_config(config, run_manager.get_child())
                )
            async for output in final_pipeline:
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
            await run_manager.on_chain_end(final)


class RunnableMapChunk(Dict[str, Any]):
    """
    Partial output from a RunnableMap
    """

    def __add__(self, other: RunnableMapChunk) -> RunnableMapChunk:
        chunk = copy.deepcopy(self)
        for key in other:
            if key not in chunk or chunk[key] is None:
                chunk[key] = other[key]
            elif other[key] is not None:
                chunk[key] += other[key]
        return chunk


class RunnableMap(Serializable, Runnable[Input, Dict[str, Any]]):
    """
    A runnable that runs a mapping of runnables in parallel,
    and returns a mapping of their outputs.
    """

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
        super().__init__(steps={key: coerce_to_runnable(r) for key, r in steps.items()})

    @property
    def lc_serializable(self) -> bool:
        return True

    @property
    def lc_namespace(self) -> List[str]:
        return self.__class__.__module__.split(".")[:-1]

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
        run_manager = callback_manager.on_chain_start(dumpd(self), input)

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
                        patch_config(config, run_manager.get_child()),
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
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
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
        run_manager = await callback_manager.on_chain_start(dumpd(self), input)

        # gather results from all steps
        try:
            # copy to avoid issues from the caller mutating the steps during invoke()
            steps = dict(self.steps)
            results = await asyncio.gather(
                *(
                    step.ainvoke(
                        input,
                        # mark each step as a child run
                        patch_config(config, run_manager.get_child()),
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

    def _transform(
        self,
        input: Iterator[Input],
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
    ) -> Iterator[RunnableMapChunk]:
        # Shallow copy steps to ignore mutations while in progress
        steps = dict(self.steps)
        # Each step gets a copy of the input iterator,
        # which is consumed in parallel in a separate thread.
        input_copies = list(safetee(input, len(steps), lock=threading.Lock()))
        with ThreadPoolExecutor() as executor:
            # Create the transform() generator for each step
            named_generators = [
                (
                    name,
                    step.transform(
                        input_copies.pop(),
                        patch_config(config, run_manager.get_child()),
                    ),
                )
                for name, step in steps.items()
            ]
            # Start the first iteration of each generator
            futures = {
                executor.submit(next, generator): (step_name, generator)
                for step_name, generator in named_generators
            }
            # Yield chunks from each as they become available,
            # and start the next iteration of that generator that yielded it.
            # When all generators are exhausted, stop.
            while futures:
                completed_futures, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in completed_futures:
                    (step_name, generator) = futures.pop(future)
                    try:
                        chunk = RunnableMapChunk({step_name: future.result()})
                        yield chunk
                        futures[executor.submit(next, generator)] = (
                            step_name,
                            generator,
                        )
                    except StopIteration:
                        pass

    def transform(
        self,
        input: Iterator[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
        yield from self._transform_stream_with_config(
            input, self._transform, config, **kwargs
        )

    def stream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Dict[str, Any]]:
        yield from self.transform(iter([input]), config)

    async def _atransform(
        self,
        input: AsyncIterator[Input],
        run_manager: AsyncCallbackManagerForChainRun,
        config: RunnableConfig,
    ) -> AsyncIterator[RunnableMapChunk]:
        # Shallow copy steps to ignore mutations while in progress
        steps = dict(self.steps)
        # Each step gets a copy of the input iterator,
        # which is consumed in parallel in a separate thread.
        input_copies = list(atee(input, len(steps), lock=asyncio.Lock()))
        # Create the transform() generator for each step
        named_generators = [
            (
                name,
                step.atransform(
                    input_copies.pop(), patch_config(config, run_manager.get_child())
                ),
            )
            for name, step in steps.items()
        ]

        # Wrap in a coroutine to satisfy linter
        async def get_next_chunk(generator: AsyncIterator) -> Optional[Output]:
            return await py_anext(generator)

        # Start the first iteration of each generator
        tasks = {
            asyncio.create_task(get_next_chunk(generator)): (step_name, generator)
            for step_name, generator in named_generators
        }
        # Yield chunks from each as they become available,
        # and start the next iteration of the generator that yielded it.
        # When all generators are exhausted, stop.
        while tasks:
            completed_tasks, _ = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in completed_tasks:
                (step_name, generator) = tasks.pop(task)
                try:
                    chunk = RunnableMapChunk({step_name: task.result()})
                    yield chunk
                    new_task = asyncio.create_task(get_next_chunk(generator))
                    tasks[new_task] = (step_name, generator)
                except StopAsyncIteration:
                    pass

    async def atransform(
        self,
        input: AsyncIterator[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        async for chunk in self._atransform_stream_with_config(
            input, self._atransform, config, **kwargs
        ):
            yield chunk

    async def astream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Dict[str, Any]]:
        async def input_aiter() -> AsyncIterator[Input]:
            yield input

        async for chunk in self.atransform(input_aiter(), config):
            yield chunk


class RunnableLambda(Runnable[Input, Output]):
    """
    A runnable that runs a callable.
    """

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

    def invoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Output:
        return self._call_with_config(self.func, input, config)


class RunnableBinding(Serializable, Runnable[Input, Output]):
    """
    A runnable that delegates calls to another runnable with a set of kwargs.
    """

    bound: Runnable[Input, Output]

    kwargs: Mapping[str, Any]

    class Config:
        arbitrary_types_allowed = True

    @property
    def lc_serializable(self) -> bool:
        return True

    @property
    def lc_namespace(self) -> List[str]:
        return self.__class__.__module__.split(".")[:-1]

    def bind(self, **kwargs: Any) -> Runnable[Input, Output]:
        return self.__class__(bound=self.bound, kwargs={**self.kwargs, **kwargs})

    def invoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Output:
        return self.bound.invoke(input, config, **{**self.kwargs, **kwargs})

    async def ainvoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Output:
        return await self.bound.ainvoke(input, config, **{**self.kwargs, **kwargs})

    def batch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        max_concurrency: Optional[int] = None,
        **kwargs: Optional[Any],
    ) -> List[Output]:
        return self.bound.batch(
            inputs, config, max_concurrency=max_concurrency, **{**self.kwargs, **kwargs}
        )

    async def abatch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        max_concurrency: Optional[int] = None,
        **kwargs: Optional[Any],
    ) -> List[Output]:
        return await self.bound.abatch(
            inputs, config, max_concurrency=max_concurrency, **{**self.kwargs, **kwargs}
        )

    def stream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Output]:
        yield from self.bound.stream(input, config, **{**self.kwargs, **kwargs})

    async def astream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Output]:
        async for item in self.bound.astream(
            input, config, **{**self.kwargs, **kwargs}
        ):
            yield item

    def transform(
        self,
        input: Iterator[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Iterator[Output]:
        yield from self.bound.transform(input, config, **{**self.kwargs, **kwargs})

    async def atransform(
        self,
        input: AsyncIterator[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Output]:
        async for item in self.bound.atransform(
            input, config, **{**self.kwargs, **kwargs}
        ):
            yield item


def patch_config(
    config: RunnableConfig, callback_manager: BaseCallbackManager
) -> RunnableConfig:
    config = config.copy()
    config["callbacks"] = callback_manager
    return config


def coerce_to_runnable(
    thing: Union[
        Runnable[Input, Output],
        Callable[[Input], Output],
        Mapping[str, Any],
    ]
) -> Runnable[Input, Output]:
    if isinstance(thing, Runnable):
        return thing
    elif callable(thing):
        return RunnableLambda(thing)
    elif isinstance(thing, dict):
        runnables: Mapping[str, Runnable[Any, Any]] = {
            key: coerce_to_runnable(r) for key, r in thing.items()
        }
        return cast(Runnable[Input, Output], RunnableMap(steps=runnables))
    else:
        raise TypeError(
            f"Expected a Runnable, callable or dict."
            f"Instead got an unsupported type: {type(thing)}"
        )
