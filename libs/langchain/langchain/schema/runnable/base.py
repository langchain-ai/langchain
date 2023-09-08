from __future__ import annotations

import asyncio
import inspect
import threading
from abc import ABC, abstractmethod
from concurrent.futures import FIRST_COMPLETED, wait
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


from langchain.load.dump import dumpd
from langchain.load.serializable import Serializable
from langchain.pydantic_v1 import Field
from langchain.schema.runnable.config import (
    RunnableConfig,
    acall_func_with_variable_args,
    call_func_with_variable_args,
    ensure_config,
    get_async_callback_manager_for_config,
    get_callback_manager_for_config,
    get_config_list,
    get_executor_for_config,
    patch_config,
)
from langchain.schema.runnable.utils import (
    Input,
    Output,
    accepts_config,
    accepts_run_manager,
    gather_with_concurrency,
)
from langchain.utils.aiter import atee, py_anext
from langchain.utils.iter import safetee

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
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> List[Output]:
        """
        Default implementation of batch, which calls invoke N times.
        Subclasses should override this method if they can batch more efficiently.
        """
        configs = get_config_list(config, len(inputs))

        def invoke(input: Input, config: RunnableConfig) -> Union[Output, Exception]:
            if return_exceptions:
                try:
                    return self.invoke(input, config, **kwargs)
                except Exception as e:
                    return e
            else:
                return self.invoke(input, config, **kwargs)

        # If there's only one input, don't bother with the executor
        if len(inputs) == 1:
            return cast(List[Output], [invoke(inputs[0], configs[0])])

        with get_executor_for_config(configs[0]) as executor:
            return cast(List[Output], list(executor.map(invoke, inputs, configs)))

    async def abatch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> List[Output]:
        """
        Default implementation of abatch, which calls ainvoke N times.
        Subclasses should override this method if they can batch more efficiently.
        """
        configs = get_config_list(config, len(inputs))

        async def ainvoke(
            input: Input, config: RunnableConfig
        ) -> Union[Output, Exception]:
            if return_exceptions:
                try:
                    return await self.ainvoke(input, config, **kwargs)
                except Exception as e:
                    return e
            else:
                return await self.ainvoke(input, config, **kwargs)

        coros = map(ainvoke, inputs, configs)
        return await gather_with_concurrency(configs[0].get("max_concurrency"), *coros)

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
        return RunnableBinding(bound=self, kwargs=kwargs, config={})

    def with_config(
        self,
        config: Optional[RunnableConfig] = None,
        # Sadly Unpack is not well supported by mypy so this will have to be untyped
        **kwargs: Any,
    ) -> Runnable[Input, Output]:
        """
        Bind config to a Runnable, returning a new Runnable.
        """
        return RunnableBinding(
            bound=self, config={**(config or {}), **kwargs}, kwargs={}
        )

    def with_retry(
        self,
        *,
        retry_if_exception_type: Tuple[Type[BaseException], ...] = (Exception,),
        wait_exponential_jitter: bool = True,
        stop_after_attempt: int = 3,
    ) -> Runnable[Input, Output]:
        from langchain.schema.runnable.retry import RunnableRetry

        return RunnableRetry(
            bound=self,
            kwargs={},
            config={},
            retry_exception_types=retry_if_exception_type,
            wait_exponential_jitter=wait_exponential_jitter,
            max_attempt_number=stop_after_attempt,
        )

    def map(self) -> Runnable[List[Input], List[Output]]:
        """
        Return a new Runnable that maps a list of inputs to a list of outputs,
        by calling invoke() with each input.
        """
        return RunnableEach(bound=self)

    def with_fallbacks(
        self,
        fallbacks: Sequence[Runnable[Input, Output]],
        *,
        exceptions_to_handle: Tuple[Type[BaseException], ...] = (Exception,),
    ) -> RunnableWithFallbacks[Input, Output]:
        return RunnableWithFallbacks(
            runnable=self,
            fallbacks=fallbacks,
            exceptions_to_handle=exceptions_to_handle,
        )

    """ --- Helper methods for Subclasses --- """

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
        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        run_manager = callback_manager.on_chain_start(
            dumpd(self),
            input,
            run_type=run_type,
            name=config.get("run_name"),
        )
        try:
            output = call_func_with_variable_args(func, input, run_manager, config)
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
        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)
        run_manager = await callback_manager.on_chain_start(
            dumpd(self),
            input,
            run_type=run_type,
            name=config.get("run_name"),
        )
        try:
            output = await acall_func_with_variable_args(
                func, input, run_manager, config
            )
        except Exception as e:
            await run_manager.on_chain_error(e)
            raise
        else:
            await run_manager.on_chain_end(dumpd(output))
            return output

    def _batch_with_config(
        self,
        func: Union[
            Callable[[List[Input]], List[Union[Exception, Output]]],
            Callable[
                [List[Input], List[CallbackManagerForChainRun]],
                List[Union[Exception, Output]],
            ],
            Callable[
                [List[Input], List[CallbackManagerForChainRun], List[RunnableConfig]],
                List[Union[Exception, Output]],
            ],
        ],
        input: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        run_type: Optional[str] = None,
    ) -> List[Output]:
        """Helper method to transform an Input value to an Output value,
        with callbacks. Use this method to implement invoke() in subclasses."""
        configs = get_config_list(config, len(input))
        callback_managers = [get_callback_manager_for_config(c) for c in configs]
        run_managers = [
            callback_manager.on_chain_start(
                dumpd(self),
                input,
                run_type=run_type,
                name=config.get("run_name"),
            )
            for callback_manager, input, config in zip(
                callback_managers, input, configs
            )
        ]
        try:
            kwargs: Dict[str, Any] = {}
            if accepts_config(func):
                kwargs["config"] = [
                    patch_config(c, callbacks=rm.get_child())
                    for c, rm in zip(configs, run_managers)
                ]
            if accepts_run_manager(func):
                kwargs["run_manager"] = run_managers
            output = func(input, **kwargs)  # type: ignore[call-arg]
        except Exception as e:
            for run_manager in run_managers:
                run_manager.on_chain_error(e)
            if return_exceptions:
                return cast(List[Output], [e for _ in input])
            else:
                raise
        else:
            first_exception: Optional[Exception] = None
            for run_manager, out in zip(run_managers, output):
                if isinstance(out, Exception):
                    first_exception = first_exception or out
                    run_manager.on_chain_error(out)
                else:
                    run_manager.on_chain_end(dumpd(out))
            if return_exceptions or first_exception is None:
                return cast(List[Output], output)
            else:
                raise first_exception

    async def _abatch_with_config(
        self,
        func: Union[
            Callable[[List[Input]], Awaitable[List[Union[Exception, Output]]]],
            Callable[
                [List[Input], List[AsyncCallbackManagerForChainRun]],
                Awaitable[List[Union[Exception, Output]]],
            ],
            Callable[
                [
                    List[Input],
                    List[AsyncCallbackManagerForChainRun],
                    List[RunnableConfig],
                ],
                Awaitable[List[Union[Exception, Output]]],
            ],
        ],
        input: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        run_type: Optional[str] = None,
    ) -> List[Output]:
        """Helper method to transform an Input value to an Output value,
        with callbacks. Use this method to implement invoke() in subclasses."""
        configs = get_config_list(config, len(input))
        callback_managers = [get_async_callback_manager_for_config(c) for c in configs]
        run_managers: List[AsyncCallbackManagerForChainRun] = await asyncio.gather(
            *(
                callback_manager.on_chain_start(
                    dumpd(self),
                    input,
                    run_type=run_type,
                    name=config.get("run_name"),
                )
                for callback_manager, input, config in zip(
                    callback_managers, input, configs
                )
            )
        )
        try:
            kwargs: Dict[str, Any] = {}
            if accepts_config(func):
                kwargs["config"] = [
                    patch_config(c, callbacks=rm.get_child())
                    for c, rm in zip(configs, run_managers)
                ]
            if accepts_run_manager(func):
                kwargs["run_manager"] = run_managers
            output = await func(input, **kwargs)  # type: ignore[call-arg]
        except Exception as e:
            await asyncio.gather(
                *(run_manager.on_chain_error(e) for run_manager in run_managers)
            )
            if return_exceptions:
                return cast(List[Output], [e for _ in input])
            else:
                raise
        else:
            first_exception: Optional[Exception] = None
            coros: List[Awaitable[None]] = []
            for run_manager, out in zip(run_managers, output):
                if isinstance(out, Exception):
                    first_exception = first_exception or out
                    coros.append(run_manager.on_chain_error(out))
                else:
                    coros.append(run_manager.on_chain_end(dumpd(out)))
            await asyncio.gather(*coros)
            if return_exceptions or first_exception is None:
                return cast(List[Output], output)
            else:
                raise first_exception

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
        # tee the input so we can iterate over it twice
        input_for_tracing, input_for_transform = tee(input, 2)
        # Start the input iterator to ensure the input runnable starts before this one
        final_input: Optional[Input] = next(input_for_tracing, None)
        final_input_supported = True
        final_output: Optional[Output] = None
        final_output_supported = True

        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        run_manager = callback_manager.on_chain_start(
            dumpd(self),
            {"input": ""},
            run_type=run_type,
            name=config.get("run_name"),
        )
        try:
            kwargs: Dict[str, Any] = {}
            if accepts_config(transformer):
                kwargs["config"] = patch_config(
                    config, callbacks=run_manager.get_child()
                )
            if accepts_run_manager(transformer):
                kwargs["run_manager"] = run_manager
            iterator = transformer(
                input_for_transform, **kwargs
            )  # type: ignore[call-arg]
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
        # tee the input so we can iterate over it twice
        input_for_tracing, input_for_transform = atee(input, 2)
        # Start the input iterator to ensure the input runnable starts before this one
        final_input: Optional[Input] = await py_anext(input_for_tracing, None)
        final_input_supported = True
        final_output: Optional[Output] = None
        final_output_supported = True

        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)
        run_manager = await callback_manager.on_chain_start(
            dumpd(self),
            {"input": ""},
            run_type=run_type,
            name=config.get("run_name"),
        )
        try:
            kwargs: Dict[str, Any] = {}
            if accepts_config(transformer):
                kwargs["config"] = patch_config(
                    config, callbacks=run_manager.get_child()
                )
            if accepts_run_manager(transformer):
                kwargs["run_manager"] = run_manager
            iterator = transformer(
                input_for_transform, **kwargs
            )  # type: ignore[call-arg]
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
    exceptions_to_handle: Tuple[Type[BaseException], ...] = (Exception,)

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
        # setup callbacks
        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        # start the root run
        run_manager = callback_manager.on_chain_start(
            dumpd(self), input, name=config.get("run_name")
        )
        first_error = None
        for runnable in self.runnables:
            try:
                output = runnable.invoke(
                    input,
                    patch_config(config, callbacks=run_manager.get_child()),
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
        # setup callbacks
        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)
        # start the root run
        run_manager = await callback_manager.on_chain_start(
            dumpd(self), input, name=config.get("run_name")
        )

        first_error = None
        for runnable in self.runnables:
            try:
                output = await runnable.ainvoke(
                    input,
                    patch_config(config, callbacks=run_manager.get_child()),
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
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> List[Output]:
        from langchain.callbacks.manager import CallbackManager

        if return_exceptions:
            raise NotImplementedError()

        # setup callbacks
        configs = get_config_list(config, len(inputs))
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
                dumpd(self),
                input if isinstance(input, dict) else {"input": input},
                name=config.get("run_name"),
            )
            for cm, input, config in zip(callback_managers, inputs, configs)
        ]

        first_error = None
        for runnable in self.runnables:
            try:
                outputs = runnable.batch(
                    inputs,
                    [
                        # each step a child run of the corresponding root run
                        patch_config(config, callbacks=rm.get_child())
                        for rm, config in zip(run_managers, configs)
                    ],
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
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> List[Output]:
        from langchain.callbacks.manager import AsyncCallbackManager

        if return_exceptions:
            raise NotImplementedError()

        # setup callbacks
        configs = get_config_list(config, len(inputs))
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
                    dumpd(self),
                    input,
                    name=config.get("run_name"),
                )
                for cm, input, config in zip(callback_managers, inputs, configs)
            )
        )

        first_error = None
        for runnable in self.runnables:
            try:
                outputs = await runnable.abatch(
                    inputs,
                    [
                        # each step a child run of the corresponding root run
                        patch_config(config, callbacks=rm.get_child())
                        for rm, config in zip(run_managers, configs)
                    ],
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
        # setup callbacks
        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        # start the root run
        run_manager = callback_manager.on_chain_start(
            dumpd(self), input, name=config.get("run_name")
        )

        # invoke all steps in sequence
        try:
            for i, step in enumerate(self.steps):
                input = step.invoke(
                    input,
                    # mark each step as a child run
                    patch_config(
                        config, callbacks=run_manager.get_child(f"seq:step:{i+1}")
                    ),
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
        # setup callbacks
        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)
        # start the root run
        run_manager = await callback_manager.on_chain_start(
            dumpd(self), input, name=config.get("run_name")
        )

        # invoke all steps in sequence
        try:
            for i, step in enumerate(self.steps):
                input = await step.ainvoke(
                    input,
                    # mark each step as a child run
                    patch_config(
                        config, callbacks=run_manager.get_child(f"seq:step:{i+1}")
                    ),
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
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> List[Output]:
        from langchain.callbacks.manager import CallbackManager

        # setup callbacks
        configs = get_config_list(config, len(inputs))
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
                dumpd(self),
                input,
                name=config.get("run_name"),
            )
            for cm, input, config in zip(callback_managers, inputs, configs)
        ]

        # invoke
        try:
            if return_exceptions:
                # Track which inputs (by index) failed so far
                # If an input has failed it will be present in this map,
                # and the value will be the exception that was raised.
                failed_inputs_map: Dict[int, Exception] = {}
                for stepidx, step in enumerate(self.steps):
                    # Assemble the original indexes of the remaining inputs
                    # (i.e. the ones that haven't failed yet)
                    remaining_idxs = [
                        i for i in range(len(configs)) if i not in failed_inputs_map
                    ]
                    # Invoke the step on the remaining inputs
                    inputs = step.batch(
                        [
                            inp
                            for i, inp in zip(remaining_idxs, inputs)
                            if i not in failed_inputs_map
                        ],
                        [
                            # each step a child run of the corresponding root run
                            patch_config(
                                config, callbacks=rm.get_child(f"seq:step:{stepidx+1}")
                            )
                            for i, (rm, config) in enumerate(zip(run_managers, configs))
                            if i not in failed_inputs_map
                        ],
                        return_exceptions=return_exceptions,
                        **kwargs,
                    )
                    # If an input failed, add it to the map
                    for i, inp in zip(remaining_idxs, inputs):
                        if isinstance(inp, Exception):
                            failed_inputs_map[i] = inp
                    inputs = [inp for inp in inputs if not isinstance(inp, Exception)]
                    # If all inputs have failed, stop processing
                    if len(failed_inputs_map) == len(configs):
                        break

                # Reassemble the outputs, inserting Exceptions for failed inputs
                inputs_copy = inputs.copy()
                inputs = []
                for i in range(len(configs)):
                    if i in failed_inputs_map:
                        inputs.append(cast(Input, failed_inputs_map[i]))
                    else:
                        inputs.append(inputs_copy.pop(0))
            else:
                for i, step in enumerate(self.steps):
                    inputs = step.batch(
                        inputs,
                        [
                            # each step a child run of the corresponding root run
                            patch_config(
                                config, callbacks=rm.get_child(f"seq:step:{i+1}")
                            )
                            for rm, config in zip(run_managers, configs)
                        ],
                    )

        # finish the root runs
        except (KeyboardInterrupt, Exception) as e:
            for rm in run_managers:
                rm.on_chain_error(e)
            if return_exceptions:
                return cast(List[Output], [e for _ in inputs])
            else:
                raise
        else:
            first_exception: Optional[Exception] = None
            for run_manager, out in zip(run_managers, inputs):
                if isinstance(out, Exception):
                    first_exception = first_exception or out
                    run_manager.on_chain_error(out)
                else:
                    run_manager.on_chain_end(dumpd(out))
            if return_exceptions or first_exception is None:
                return cast(List[Output], inputs)
            else:
                raise first_exception

    async def abatch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> List[Output]:
        from langchain.callbacks.manager import (
            AsyncCallbackManager,
        )

        # setup callbacks
        configs = get_config_list(config, len(inputs))
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
                    dumpd(self),
                    input,
                    name=config.get("run_name"),
                )
                for cm, input, config in zip(callback_managers, inputs, configs)
            )
        )

        # invoke .batch() on each step
        # this uses batching optimizations in Runnable subclasses, like LLM
        try:
            if return_exceptions:
                # Track which inputs (by index) failed so far
                # If an input has failed it will be present in this map,
                # and the value will be the exception that was raised.
                failed_inputs_map: Dict[int, Exception] = {}
                for stepidx, step in enumerate(self.steps):
                    # Assemble the original indexes of the remaining inputs
                    # (i.e. the ones that haven't failed yet)
                    remaining_idxs = [
                        i for i in range(len(configs)) if i not in failed_inputs_map
                    ]
                    # Invoke the step on the remaining inputs
                    inputs = await step.abatch(
                        [
                            inp
                            for i, inp in zip(remaining_idxs, inputs)
                            if i not in failed_inputs_map
                        ],
                        [
                            # each step a child run of the corresponding root run
                            patch_config(
                                config, callbacks=rm.get_child(f"seq:step:{stepidx+1}")
                            )
                            for i, (rm, config) in enumerate(zip(run_managers, configs))
                            if i not in failed_inputs_map
                        ],
                        return_exceptions=return_exceptions,
                        **kwargs,
                    )
                    # If an input failed, add it to the map
                    for i, inp in zip(remaining_idxs, inputs):
                        if isinstance(inp, Exception):
                            failed_inputs_map[i] = inp
                    inputs = [inp for inp in inputs if not isinstance(inp, Exception)]
                    # If all inputs have failed, stop processing
                    if len(failed_inputs_map) == len(configs):
                        break

                # Reassemble the outputs, inserting Exceptions for failed inputs
                inputs_copy = inputs.copy()
                inputs = []
                for i in range(len(configs)):
                    if i in failed_inputs_map:
                        inputs.append(cast(Input, failed_inputs_map[i]))
                    else:
                        inputs.append(inputs_copy.pop(0))
            else:
                for i, step in enumerate(self.steps):
                    inputs = await step.abatch(
                        inputs,
                        [
                            # each step a child run of the corresponding root run
                            patch_config(
                                config, callbacks=rm.get_child(f"seq:step:{i+1}")
                            )
                            for rm, config in zip(run_managers, configs)
                        ],
                    )
        # finish the root runs
        except (KeyboardInterrupt, Exception) as e:
            await asyncio.gather(*(rm.on_chain_error(e) for rm in run_managers))
            if return_exceptions:
                return cast(List[Output], [e for _ in inputs])
            else:
                raise
        else:
            first_exception: Optional[Exception] = None
            coros: List[Awaitable[None]] = []
            for run_manager, out in zip(run_managers, inputs):
                if isinstance(out, Exception):
                    first_exception = first_exception or out
                    coros.append(run_manager.on_chain_error(out))
                else:
                    coros.append(run_manager.on_chain_end(dumpd(out)))
            await asyncio.gather(*coros)
            if return_exceptions or first_exception is None:
                return cast(List[Output], inputs)
            else:
                raise first_exception

    def stream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Output]:
        # setup callbacks
        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        # start the root run
        run_manager = callback_manager.on_chain_start(
            dumpd(self), input, name=config.get("run_name")
        )

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
                    patch_config(
                        config,
                        callbacks=run_manager.get_child(
                            f"seq:step:{steps.index(step)+1}"
                        ),
                    ),
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
                input,
                patch_config(
                    config,
                    callbacks=run_manager.get_child(
                        f"seq:step:{streaming_start_index+1}"
                    ),
                ),
            )
            # stream the rest of the last steps with streaming input
            for step in steps[streaming_start_index + 1 :]:
                final_pipeline = step.transform(
                    final_pipeline,
                    patch_config(
                        config,
                        callbacks=run_manager.get_child(
                            f"seq:step:{steps.index(step)+1}"
                        ),
                    ),
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
        # setup callbacks
        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)
        # start the root run
        run_manager = await callback_manager.on_chain_start(
            dumpd(self), input, name=config.get("run_name")
        )

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
                    patch_config(
                        config,
                        callbacks=run_manager.get_child(
                            f"seq:step:{steps.index(step)+1}"
                        ),
                    ),
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
                input,
                patch_config(
                    config,
                    callbacks=run_manager.get_child(
                        f"seq:step:{streaming_start_index+1}"
                    ),
                ),
            )
            # stream the rest of the last steps with streaming input
            for step in steps[streaming_start_index + 1 :]:
                final_pipeline = step.atransform(
                    final_pipeline,
                    patch_config(
                        config,
                        callbacks=run_manager.get_child(
                            f"seq:step:{steps.index(step)+1}"
                        ),
                    ),
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
        chunk = RunnableMapChunk(self)
        for key in other:
            if key not in chunk or chunk[key] is None:
                chunk[key] = other[key]
            elif other[key] is not None:
                chunk[key] += other[key]
        return chunk

    def __radd__(self, other: RunnableMapChunk) -> RunnableMapChunk:
        chunk = RunnableMapChunk(other)
        for key in self:
            if key not in chunk or chunk[key] is None:
                chunk[key] = self[key]
            elif self[key] is not None:
                chunk[key] += self[key]
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
        config = ensure_config(config)
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
            dumpd(self), input, name=config.get("run_name")
        )

        # gather results from all steps
        try:
            # copy to avoid issues from the caller mutating the steps during invoke()
            steps = dict(self.steps)
            with get_executor_for_config(config) as executor:
                futures = [
                    executor.submit(
                        step.invoke,
                        input,
                        # mark each step as a child run
                        patch_config(
                            config,
                            deep_copy_locals=True,
                            callbacks=run_manager.get_child(f"map:key:{key}"),
                        ),
                    )
                    for key, step in steps.items()
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
        # setup callbacks
        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)
        # start the root run
        run_manager = await callback_manager.on_chain_start(
            dumpd(self), input, name=config.get("run_name")
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
                        patch_config(
                            config, callbacks=run_manager.get_child(f"map:key:{key}")
                        ),
                    )
                    for key, step in steps.items()
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
        with get_executor_for_config(config) as executor:
            # Create the transform() generator for each step
            named_generators = [
                (
                    name,
                    step.transform(
                        input_copies.pop(),
                        patch_config(
                            config, callbacks=run_manager.get_child(f"map:key:{name}")
                        ),
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
                    input_copies.pop(),
                    patch_config(
                        config, callbacks=run_manager.get_child(f"map:key:{name}")
                    ),
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

    def __init__(
        self,
        func: Union[Callable[[Input], Output], Callable[[Input], Awaitable[Output]]],
        afunc: Optional[Callable[[Input], Awaitable[Output]]] = None,
    ) -> None:
        if afunc is not None:
            self.afunc = afunc

        if inspect.iscoroutinefunction(func):
            self.afunc = func
        elif callable(func):
            self.func = cast(Callable[[Input], Output], func)
        else:
            raise TypeError(
                "Expected a callable type for `func`."
                f"Instead got an unsupported type: {type(func)}"
            )

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, RunnableLambda):
            if hasattr(self, "func") and hasattr(other, "func"):
                return self.func == other.func
            elif hasattr(self, "afunc") and hasattr(other, "afunc"):
                return self.afunc == other.afunc
            else:
                return False
        else:
            return False

    def __repr__(self) -> str:
        return "RunnableLambda(...)"

    def _invoke(
        self,
        input: Input,
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
    ) -> Output:
        output = call_func_with_variable_args(self.func, input, run_manager, config)
        # If the output is a runnable, invoke it
        if isinstance(output, Runnable):
            recursion_limit = config["recursion_limit"]
            if recursion_limit <= 0:
                raise RecursionError(
                    f"Recursion limit reached when invoking {self} with input {input}."
                )
            output = output.invoke(
                input,
                patch_config(
                    config,
                    callbacks=run_manager.get_child(),
                    recursion_limit=recursion_limit - 1,
                ),
            )
        return output

    async def _ainvoke(
        self,
        input: Input,
        run_manager: AsyncCallbackManagerForChainRun,
        config: RunnableConfig,
    ) -> Output:
        output = await acall_func_with_variable_args(
            self.afunc, input, run_manager, config
        )
        # If the output is a runnable, invoke it
        if isinstance(output, Runnable):
            recursion_limit = config["recursion_limit"]
            if recursion_limit <= 0:
                raise RecursionError(
                    f"Recursion limit reached when invoking {self} with input {input}."
                )
            output = await output.ainvoke(
                input,
                patch_config(
                    config,
                    callbacks=run_manager.get_child(),
                    recursion_limit=recursion_limit - 1,
                ),
            )
        return output

    def _config(
        self, config: Optional[RunnableConfig], callable: Callable[..., Any]
    ) -> RunnableConfig:
        config = config or {}

        if config.get("run_name") is None:
            try:
                run_name = callable.__name__
            except AttributeError:
                run_name = None
            if run_name is not None:
                return patch_config(config, run_name=run_name)

        return config

    def invoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Output:
        if hasattr(self, "func"):
            return self._call_with_config(
                self._invoke,
                input,
                self._config(config, self.func),
            )
        else:
            raise TypeError(
                "Cannot invoke a coroutine function synchronously."
                "Use `ainvoke` instead."
            )

    async def ainvoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Output:
        if hasattr(self, "afunc"):
            return await self._acall_with_config(
                self._ainvoke,
                input,
                self._config(config, self.afunc),
            )
        else:
            # Delegating to super implementation of ainvoke.
            # Uses asyncio executor to run the sync version (invoke)
            return await super().ainvoke(input, config)


class RunnableEach(Serializable, Runnable[List[Input], List[Output]]):
    """
    A runnable that delegates calls to another runnable
    with each element of the input sequence.
    """

    bound: Runnable[Input, Output]

    class Config:
        arbitrary_types_allowed = True

    @property
    def lc_serializable(self) -> bool:
        return True

    @property
    def lc_namespace(self) -> List[str]:
        return self.__class__.__module__.split(".")[:-1]

    def bind(self, **kwargs: Any) -> RunnableEach[Input, Output]:
        return RunnableEach(bound=self.bound.bind(**kwargs))

    def _invoke(
        self,
        inputs: List[Input],
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
    ) -> List[Output]:
        return self.bound.batch(
            inputs, patch_config(config, callbacks=run_manager.get_child())
        )

    def invoke(
        self, input: List[Input], config: Optional[RunnableConfig] = None
    ) -> List[Output]:
        return self._call_with_config(self._invoke, input, config)

    async def _ainvoke(
        self,
        inputs: List[Input],
        run_manager: AsyncCallbackManagerForChainRun,
        config: RunnableConfig,
    ) -> List[Output]:
        return await self.bound.abatch(
            inputs, patch_config(config, callbacks=run_manager.get_child())
        )

    async def ainvoke(
        self, input: List[Input], config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> List[Output]:
        return await self._acall_with_config(self._ainvoke, input, config)


class RunnableBinding(Serializable, Runnable[Input, Output]):
    """
    A runnable that delegates calls to another runnable with a set of kwargs.
    """

    bound: Runnable[Input, Output]

    kwargs: Mapping[str, Any]

    config: Mapping[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    @property
    def lc_serializable(self) -> bool:
        return True

    @property
    def lc_namespace(self) -> List[str]:
        return self.__class__.__module__.split(".")[:-1]

    def _merge_config(self, config: Optional[RunnableConfig]) -> RunnableConfig:
        copy = cast(RunnableConfig, dict(self.config))
        if config:
            for key in config:
                # Even though the keys aren't literals this is correct
                # because both dicts are same type
                copy[key] = config[key] or copy.get(key)  # type: ignore
        return copy

    def bind(self, **kwargs: Any) -> Runnable[Input, Output]:
        return self.__class__(
            bound=self.bound, config=self.config, kwargs={**self.kwargs, **kwargs}
        )

    def with_config(
        self,
        config: Optional[RunnableConfig] = None,
        # Sadly Unpack is not well supported by mypy so this will have to be untyped
        **kwargs: Any,
    ) -> Runnable[Input, Output]:
        return self.__class__(
            bound=self.bound,
            kwargs=self.kwargs,
            config={**self.config, **(config or {}), **kwargs},
        )

    def with_retry(self, **kwargs: Any) -> Runnable[Input, Output]:
        return self.__class__(
            bound=self.bound.with_retry(**kwargs),
            kwargs=self.kwargs,
            config=self.config,
        )

    def invoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Output:
        return self.bound.invoke(
            input,
            self._merge_config(config),
            **{**self.kwargs, **kwargs},
        )

    async def ainvoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Output:
        return await self.bound.ainvoke(
            input,
            self._merge_config(config),
            **{**self.kwargs, **kwargs},
        )

    def batch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> List[Output]:
        if isinstance(config, list):
            configs = cast(
                List[RunnableConfig], [self._merge_config(conf) for conf in config]
            )
        else:
            configs = [
                patch_config(self._merge_config(config), deep_copy_locals=True)
                for _ in range(len(inputs))
            ]
        return self.bound.batch(
            inputs,
            configs,
            return_exceptions=return_exceptions,
            **{**self.kwargs, **kwargs},
        )

    async def abatch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> List[Output]:
        if isinstance(config, list):
            configs = cast(
                List[RunnableConfig], [self._merge_config(conf) for conf in config]
            )
        else:
            configs = [
                patch_config(self._merge_config(config), deep_copy_locals=True)
                for _ in range(len(inputs))
            ]
        return await self.bound.abatch(
            inputs,
            configs,
            return_exceptions=return_exceptions,
            **{**self.kwargs, **kwargs},
        )

    def stream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Output]:
        yield from self.bound.stream(
            input,
            self._merge_config(config),
            **{**self.kwargs, **kwargs},
        )

    async def astream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Output]:
        async for item in self.bound.astream(
            input,
            self._merge_config(config),
            **{**self.kwargs, **kwargs},
        ):
            yield item

    def transform(
        self,
        input: Iterator[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Iterator[Output]:
        yield from self.bound.transform(
            input,
            self._merge_config(config),
            **{**self.kwargs, **kwargs},
        )

    async def atransform(
        self,
        input: AsyncIterator[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Output]:
        async for item in self.bound.atransform(
            input,
            self._merge_config(config),
            **{**self.kwargs, **kwargs},
        ):
            yield item


RunnableBinding.update_forward_refs(RunnableConfig=RunnableConfig)


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
