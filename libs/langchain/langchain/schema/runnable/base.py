from __future__ import annotations

import asyncio
import inspect
import threading
from abc import ABC, abstractmethod
from concurrent.futures import FIRST_COMPLETED, wait
from functools import partial
from itertools import tee
from operator import itemgetter
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
    overload,
)

from typing_extensions import Literal, get_args

if TYPE_CHECKING:
    from langchain.schema.callbacks.manager import (
        AsyncCallbackManagerForChainRun,
        CallbackManagerForChainRun,
    )
    from langchain.schema.callbacks.tracers.log_stream import RunLog, RunLogPatch
    from langchain.schema.callbacks.tracers.root_listeners import Listener
    from langchain.schema.runnable.fallbacks import (
        RunnableWithFallbacks as RunnableWithFallbacksT,
    )

from langchain.load.dump import dumpd
from langchain.load.serializable import Serializable
from langchain.pydantic_v1 import BaseModel, Field, create_model
from langchain.schema.runnable.config import (
    RunnableConfig,
    acall_func_with_variable_args,
    call_func_with_variable_args,
    ensure_config,
    get_async_callback_manager_for_config,
    get_callback_manager_for_config,
    get_config_list,
    get_executor_for_config,
    merge_configs,
    patch_config,
)
from langchain.schema.runnable.utils import (
    AddableDict,
    AnyConfigurableField,
    ConfigurableField,
    ConfigurableFieldSpec,
    Input,
    Output,
    accepts_config,
    accepts_run_manager,
    gather_with_concurrency,
    get_function_first_arg_dict_keys,
    get_lambda_source,
    get_unique_config_specs,
    indent_lines_after_first,
)
from langchain.utils.aiter import atee, py_anext
from langchain.utils.iter import safetee

Other = TypeVar("Other")


class Runnable(Generic[Input, Output], ABC):
    """A unit of work that can be invoked, batched, streamed, transformed and composed.

     Key Methods
     ===========

    * invoke/ainvoke: Transforms a single input into an output.
    * batch/abatch: Efficiently transforms multiple inputs into outputs.
    * stream/astream: Streams output from a single input as it's produced.
    * astream_log: Streams output and selected intermediate results from an input.

    Built-in optimizations:

    * Batch: By default, batch runs invoke() in parallel using a thread pool executor.
             Override to optimize batching.

    * Async: Methods with "a" suffix are asynchronous. By default, they execute
             the sync counterpart using asyncio's thread pool.
             Override for native async.

    All methods accept an optional config argument, which can be used to configure
    execution, add tags and metadata for tracing and debugging etc.

    Runnables expose schematic information about their input, output and config via
    the input_schema property, the output_schema property and config_schema method.

    LCEL and Composition
    ====================

    The LangChain Expression Language (LCEL) is a declarative way to compose Runnables
    into chains. Any chain constructed this way will automatically have sync, async,
    batch, and streaming support.

    The main composition primitives are RunnableSequence and RunnableParallel.

    RunnableSequence invokes a series of runnables sequentially, with one runnable's
    output serving as the next's input. Construct using the `|` operator or by
    passing a list of runnables to RunnableSequence.

    RunnableParallel invokes runnables concurrently, providing the same input
    to each. Construct it using a dict literal within a sequence or by passing a
    dict to RunnableParallel.


    For example,

    .. code-block:: python

        from langchain.schema.runnable import RunnableLambda

        # A RunnableSequence constructed using the `|` operator
        sequence = RunnableLambda(lambda x: x + 1) | RunnableLambda(lambda x: x * 2)
        sequence.invoke(1) # 4
        sequence.batch([1, 2, 3]) # [4, 6, 8]


        # A sequence that contains a RunnableParallel constructed using a dict literal
        sequence = RunnableLambda(lambda x: x + 1) | {
            'mul_2': RunnableLambda(lambda x: x * 2),
            'mul_5': RunnableLambda(lambda x: x * 5)
        }
        sequence.invoke(1) # {'mul_2': 4, 'mul_5': 10}

    Standard Methods
    ================

    All Runnables expose additional methods that can be used to modify their behavior
    (e.g., add a retry policy, add lifecycle listeners, make them configurable, etc.).

    These methods will work on any Runnable, including Runnable chains constructed
    by composing other Runnables. See the individual methods for details.

    For example,

    .. code-block:: python

        from langchain.schema.runnable import RunnableLambda

        import random

        def add_one(x: int) -> int:
            return x + 1


        def buggy_double(y: int) -> int:
            '''Buggy code that will fail 70% of the time'''
            if random.random() > 0.3:
                print('This code failed, and will probably be retried!')
                raise ValueError('Triggered buggy code')
            return y * 2

        sequence = (
            RunnableLambda(add_one) |
            RunnableLambda(buggy_double).with_retry( # Retry on failure
                stop_after_attempt=10,
                wait_exponential_jitter=False
            )
        )

        print(sequence.input_schema.schema()) # Show inferred input schema
        print(sequence.output_schema.schema()) # Show inferred output schema
        print(sequence.invoke(2)) # invoke the sequence (note the retry above!!)

    Debugging and tracing
    =====================

    As the chains get longer, it can be useful to be able to see intermediate results
    to debug and trace the chain.

    You can set the global debug flag to True to enable debug output for all chains:

        .. code-block:: python

            from langchain.globals import set_debug
            set_debug(True)

    Alternatively, you can pass existing or custom callbacks to any given chain:

       ... code-block:: python

            from langchain.callbacks.tracers import ConsoleCallbackHandler

            chain.invoke(
                ...,
                config={'callbacks': [ConsoleCallbackHandler()]}
            )

    For a UI (and much more) checkout LangSmith: https://docs.smith.langchain.com/
    """

    @property
    def InputType(self) -> Type[Input]:
        """The type of input this runnable accepts specified as a type annotation."""
        for cls in self.__class__.__orig_bases__:  # type: ignore[attr-defined]
            type_args = get_args(cls)
            if type_args and len(type_args) == 2:
                return type_args[0]

        raise TypeError(
            f"Runnable {self.__class__.__name__} doesn't have an inferable InputType. "
            "Override the InputType property to specify the input type."
        )

    @property
    def OutputType(self) -> Type[Output]:
        """The type of output this runnable produces specified as a type annotation."""
        for cls in self.__class__.__orig_bases__:  # type: ignore[attr-defined]
            type_args = get_args(cls)
            if type_args and len(type_args) == 2:
                return type_args[1]

        raise TypeError(
            f"Runnable {self.__class__.__name__} doesn't have an inferable OutputType. "
            "Override the OutputType property to specify the output type."
        )

    @property
    def input_schema(self) -> Type[BaseModel]:
        """The type of input this runnable accepts specified as a pydantic model."""
        return self.get_input_schema()

    def get_input_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> Type[BaseModel]:
        """Get a pydantic model that can be used to validate input to the runnable.

        Runnables that leverage the configurable_fields and configurable_alternatives
        methods will have a dynamic input schema that depends on which
        configuration the runnable is invoked with.

        This method allows to get an input schema for a specific configuration.

        Args:
            config: A config to use when generating the schema.

        Returns:
            A pydantic model that can be used to validate input.
        """
        root_type = self.InputType

        if inspect.isclass(root_type) and issubclass(root_type, BaseModel):
            return root_type

        return create_model(
            self.__class__.__name__ + "Input", __root__=(root_type, None)
        )

    @property
    def output_schema(self) -> Type[BaseModel]:
        """The type of output this runnable produces specified as a pydantic model."""
        return self.get_output_schema()

    def get_output_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> Type[BaseModel]:
        """Get a pydantic model that can be used to validate output to the runnable.

        Runnables that leverage the configurable_fields and configurable_alternatives
        methods will have a dynamic output schema that depends on which
        configuration the runnable is invoked with.

        This method allows to get an output schema for a specific configuration.

        Args:
            config: A config to use when generating the schema.

        Returns:
            A pydantic model that can be used to validate output.
        """
        root_type = self.OutputType

        if inspect.isclass(root_type) and issubclass(root_type, BaseModel):
            return root_type

        return create_model(
            self.__class__.__name__ + "Output", __root__=(root_type, None)
        )

    @property
    def config_specs(self) -> List[ConfigurableFieldSpec]:
        """List configurable fields for this runnable."""
        return []

    def config_schema(
        self, *, include: Optional[Sequence[str]] = None
    ) -> Type[BaseModel]:
        """The type of config this runnable accepts specified as a pydantic model.

        To mark a field as configurable, see the `configurable_fields`
        and `configurable_alternatives` methods.

        Args:
            include: A list of fields to include in the config schema.

        Returns:
            A pydantic model that can be used to validate config.
        """

        class _Config:
            arbitrary_types_allowed = True

        include = include or []
        config_specs = self.config_specs
        configurable = (
            create_model(  # type: ignore[call-overload]
                "Configurable",
                **{
                    spec.id: (
                        spec.annotation,
                        Field(
                            spec.default, title=spec.name, description=spec.description
                        ),
                    )
                    for spec in config_specs
                },
            )
            if config_specs
            else None
        )

        return create_model(  # type: ignore[call-overload]
            self.__class__.__name__ + "Config",
            __config__=_Config,
            **({"configurable": (configurable, None)} if configurable else {}),
            **{
                field_name: (field_type, None)
                for field_name, field_type in RunnableConfig.__annotations__.items()
                if field_name in [i for i in include if i != "configurable"]
            },
        )

    def __or__(
        self,
        other: Union[
            Runnable[Any, Other],
            Callable[[Any], Other],
            Callable[[Iterator[Any]], Iterator[Other]],
            Mapping[str, Union[Runnable[Any, Other], Callable[[Any], Other], Any]],
        ],
    ) -> RunnableSerializable[Input, Other]:
        """Compose this runnable with another object to create a RunnableSequence."""
        return RunnableSequence(first=self, last=coerce_to_runnable(other))

    def __ror__(
        self,
        other: Union[
            Runnable[Other, Any],
            Callable[[Other], Any],
            Callable[[Iterator[Other]], Iterator[Any]],
            Mapping[str, Union[Runnable[Other, Any], Callable[[Other], Any], Any]],
        ],
    ) -> RunnableSerializable[Other, Output]:
        """Compose this runnable with another object to create a RunnableSequence."""
        return RunnableSequence(first=coerce_to_runnable(other), last=self)

    """ --- Public API --- """

    @abstractmethod
    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        """Transform a single input into an output. Override to implement.

        Args:
            input: The input to the runnable.
            config: A config to use when invoking the runnable.
               The config supports standard keys like 'tags', 'metadata' for tracing
               purposes, 'max_concurrency' for controlling how much work to do
               in parallel, and other keys. Please refer to the RunnableConfig
               for more details.

        Returns:
            The output of the runnable.
        """

    async def ainvoke(
        self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Output:
        """Default implementation of ainvoke, calls invoke from a thread.

        The default implementation allows usage of async code even if
        the runnable did not implement a native async version of invoke.

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
        """Default implementation runs invoke in parallel using a thread pool executor.

        The default implementation of batch works well for IO bound runnables.

        Subclasses should override this method if they can batch more efficiently;
        e.g., if the underlying runnable uses an API which supports a batch mode.
        """
        if not inputs:
            return []

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
        """Default implementation runs ainvoke in parallel using asyncio.gather.

        The default implementation of batch works well for IO bound runnables.

        Subclasses should override this method if they can batch more efficiently;
        e.g., if the underlying runnable uses an API which supports a batch mode.
        """
        if not inputs:
            return []

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

    @overload
    def astream_log(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        *,
        diff: Literal[True] = True,
        include_names: Optional[Sequence[str]] = None,
        include_types: Optional[Sequence[str]] = None,
        include_tags: Optional[Sequence[str]] = None,
        exclude_names: Optional[Sequence[str]] = None,
        exclude_types: Optional[Sequence[str]] = None,
        exclude_tags: Optional[Sequence[str]] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[RunLogPatch]:
        ...

    @overload
    def astream_log(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        *,
        diff: Literal[False],
        include_names: Optional[Sequence[str]] = None,
        include_types: Optional[Sequence[str]] = None,
        include_tags: Optional[Sequence[str]] = None,
        exclude_names: Optional[Sequence[str]] = None,
        exclude_types: Optional[Sequence[str]] = None,
        exclude_tags: Optional[Sequence[str]] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[RunLog]:
        ...

    async def astream_log(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        *,
        diff: bool = True,
        include_names: Optional[Sequence[str]] = None,
        include_types: Optional[Sequence[str]] = None,
        include_tags: Optional[Sequence[str]] = None,
        exclude_names: Optional[Sequence[str]] = None,
        exclude_types: Optional[Sequence[str]] = None,
        exclude_tags: Optional[Sequence[str]] = None,
        **kwargs: Optional[Any],
    ) -> Union[AsyncIterator[RunLogPatch], AsyncIterator[RunLog]]:
        """
        Stream all output from a runnable, as reported to the callback system.
        This includes all inner runs of LLMs, Retrievers, Tools, etc.

        Output is streamed as Log objects, which include a list of
        jsonpatch ops that describe how the state of the run has changed in each
        step, and the final state of the run.

        The jsonpatch ops can be applied in order to construct state.
        """

        from langchain.callbacks.base import BaseCallbackManager
        from langchain.callbacks.tracers.log_stream import (
            LogStreamCallbackHandler,
            RunLog,
            RunLogPatch,
        )

        # Create a stream handler that will emit Log objects
        stream = LogStreamCallbackHandler(
            auto_close=False,
            include_names=include_names,
            include_types=include_types,
            include_tags=include_tags,
            exclude_names=exclude_names,
            exclude_types=exclude_types,
            exclude_tags=exclude_tags,
        )

        # Assign the stream handler to the config
        config = config or {}
        callbacks = config.get("callbacks")
        if callbacks is None:
            config["callbacks"] = [stream]
        elif isinstance(callbacks, list):
            config["callbacks"] = callbacks + [stream]
        elif isinstance(callbacks, BaseCallbackManager):
            callbacks = callbacks.copy()
            callbacks.add_handler(stream, inherit=True)
            config["callbacks"] = callbacks
        else:
            raise ValueError(
                f"Unexpected type for callbacks: {callbacks}."
                "Expected None, list or AsyncCallbackManager."
            )

        # Call the runnable in streaming mode,
        # add each chunk to the output stream
        async def consume_astream() -> None:
            try:
                async for chunk in self.astream(input, config, **kwargs):
                    await stream.send_stream.send(
                        RunLogPatch(
                            {
                                "op": "add",
                                "path": "/streamed_output/-",
                                "value": chunk,
                            }
                        )
                    )
            finally:
                await stream.send_stream.aclose()

        # Start the runnable in a task, so we can start consuming output
        task = asyncio.create_task(consume_astream())

        try:
            # Yield each chunk from the output stream
            if diff:
                async for log in stream:
                    yield log
            else:
                state = RunLog(state=None)  # type: ignore[arg-type]
                async for log in stream:
                    state = state + log
                    yield state
        finally:
            # Wait for the runnable to finish, if not cancelled (eg. by break)
            try:
                await task
            except asyncio.CancelledError:
                pass

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
        final: Input
        got_first_val = False

        for chunk in input:
            if not got_first_val:
                final = chunk
                got_first_val = True
            else:
                # Make a best effort to gather, for any type that supports `+`
                # This method should throw an error if gathering fails.
                final = final + chunk  # type: ignore[operator]

        if got_first_val:
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
        final: Input
        got_first_val = False

        async for chunk in input:
            if not got_first_val:
                final = chunk
                got_first_val = True
            else:
                # Make a best effort to gather, for any type that supports `+`
                # This method should throw an error if gathering fails.
                final = final + chunk  # type: ignore[operator]

        if got_first_val:
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
            bound=self,
            config=cast(
                RunnableConfig,
                {**(config or {}), **kwargs},
            ),  # type: ignore[misc]
            kwargs={},
        )

    def with_listeners(
        self,
        *,
        on_start: Optional[Listener] = None,
        on_end: Optional[Listener] = None,
        on_error: Optional[Listener] = None,
    ) -> Runnable[Input, Output]:
        """
        Bind lifecycle listeners to a Runnable, returning a new Runnable.

        on_start: Called before the runnable starts running, with the Run object.
        on_end: Called after the runnable finishes running, with the Run object.
        on_error: Called if the runnable throws an error, with the Run object.

        The Run object contains information about the run, including its id,
        type, input, output, error, start_time, end_time, and any tags or metadata
        added to the run.
        """
        from langchain.callbacks.tracers.root_listeners import RootListenersTracer

        return RunnableBinding(
            bound=self,
            config_factories=[
                lambda config: {
                    "callbacks": [
                        RootListenersTracer(
                            config=config,
                            on_start=on_start,
                            on_end=on_end,
                            on_error=on_error,
                        )
                    ],
                }
            ],
        )

    def with_types(
        self,
        *,
        input_type: Optional[Type[Input]] = None,
        output_type: Optional[Type[Output]] = None,
    ) -> Runnable[Input, Output]:
        """
        Bind input and output types to a Runnable, returning a new Runnable.
        """
        return RunnableBinding(
            bound=self,
            custom_input_type=input_type,
            custom_output_type=output_type,
            kwargs={},
        )

    def with_retry(
        self,
        *,
        retry_if_exception_type: Tuple[Type[BaseException], ...] = (Exception,),
        wait_exponential_jitter: bool = True,
        stop_after_attempt: int = 3,
    ) -> Runnable[Input, Output]:
        """Create a new Runnable that retries the original runnable on exceptions.

        Args:
            retry_if_exception_type: A tuple of exception types to retry on
            wait_exponential_jitter: Whether to add jitter to the wait time
                                     between retries
            stop_after_attempt: The maximum number of attempts to make before giving up

        Returns:
            A new Runnable that retries the original runnable on exceptions.
        """
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
    ) -> RunnableWithFallbacksT[Input, Output]:
        """Add fallbacks to a runnable, returning a new Runnable.

        Args:
            fallbacks: A sequence of runnables to try if the original runnable fails.
            exceptions_to_handle: A tuple of exception types to handle.

        Returns:
            A new Runnable that will try the original runnable, and then each
            fallback in order, upon failures.
        """
        from langchain.schema.runnable.fallbacks import RunnableWithFallbacks

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
        **kwargs: Optional[Any],
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
            output = call_func_with_variable_args(
                func, input, config, run_manager, **kwargs
            )
        except BaseException as e:
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
        **kwargs: Optional[Any],
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
                func, input, config, run_manager, **kwargs
            )
        except BaseException as e:
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
        **kwargs: Optional[Any],
    ) -> List[Output]:
        """Helper method to transform an Input value to an Output value,
        with callbacks. Use this method to implement invoke() in subclasses."""
        if not input:
            return []

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
            if accepts_config(func):
                kwargs["config"] = [
                    patch_config(c, callbacks=rm.get_child())
                    for c, rm in zip(configs, run_managers)
                ]
            if accepts_run_manager(func):
                kwargs["run_manager"] = run_managers
            output = func(input, **kwargs)  # type: ignore[call-arg]
        except BaseException as e:
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
        **kwargs: Optional[Any],
    ) -> List[Output]:
        """Helper method to transform an Input value to an Output value,
        with callbacks. Use this method to implement invoke() in subclasses."""
        if not input:
            return []

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
            if accepts_config(func):
                kwargs["config"] = [
                    patch_config(c, callbacks=rm.get_child())
                    for c, rm in zip(configs, run_managers)
                ]
            if accepts_run_manager(func):
                kwargs["run_manager"] = run_managers
            output = await func(input, **kwargs)  # type: ignore[call-arg]
        except BaseException as e:
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
        **kwargs: Optional[Any],
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
            if accepts_config(transformer):
                kwargs["config"] = patch_config(
                    config, callbacks=run_manager.get_child()
                )
            if accepts_run_manager(transformer):
                kwargs["run_manager"] = run_manager
            iterator = transformer(input_for_transform, **kwargs)  # type: ignore[call-arg]
            for chunk in iterator:
                yield chunk
                if final_output_supported:
                    if final_output is None:
                        final_output = chunk
                    else:
                        try:
                            final_output = final_output + chunk  # type: ignore
                        except TypeError:
                            final_output = None
                            final_output_supported = False
            for ichunk in input_for_tracing:
                if final_input_supported:
                    if final_input is None:
                        final_input = ichunk
                    else:
                        try:
                            final_input = final_input + ichunk  # type: ignore
                        except TypeError:
                            final_input = None
                            final_input_supported = False
        except BaseException as e:
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
        **kwargs: Optional[Any],
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
            if accepts_config(transformer):
                kwargs["config"] = patch_config(
                    config, callbacks=run_manager.get_child()
                )
            if accepts_run_manager(transformer):
                kwargs["run_manager"] = run_manager
            iterator = transformer(input_for_transform, **kwargs)  # type: ignore[call-arg]
            async for chunk in iterator:
                yield chunk
                if final_output_supported:
                    if final_output is None:
                        final_output = chunk
                    else:
                        try:
                            final_output = final_output + chunk  # type: ignore
                        except TypeError:
                            final_output = None
                            final_output_supported = False
            async for ichunk in input_for_tracing:
                if final_input_supported:
                    if final_input is None:
                        final_input = ichunk
                    else:
                        try:
                            final_input = final_input + ichunk  # type: ignore[operator]
                        except TypeError:
                            final_input = None
                            final_input_supported = False
        except BaseException as e:
            await run_manager.on_chain_error(e, inputs=final_input)
            raise
        else:
            await run_manager.on_chain_end(final_output, inputs=final_input)


class RunnableSerializable(Serializable, Runnable[Input, Output]):
    """A Runnable that can be serialized to JSON."""

    def configurable_fields(
        self, **kwargs: AnyConfigurableField
    ) -> RunnableSerializable[Input, Output]:
        from langchain.schema.runnable.configurable import RunnableConfigurableFields

        for key in kwargs:
            if key not in self.__fields__:
                raise ValueError(
                    f"Configuration key {key} not found in {self}: "
                    "available keys are {self.__fields__.keys()}"
                )

        return RunnableConfigurableFields(default=self, fields=kwargs)

    def configurable_alternatives(
        self,
        which: ConfigurableField,
        default_key: str = "default",
        **kwargs: Union[Runnable[Input, Output], Callable[[], Runnable[Input, Output]]],
    ) -> RunnableSerializable[Input, Output]:
        from langchain.schema.runnable.configurable import (
            RunnableConfigurableAlternatives,
        )

        return RunnableConfigurableAlternatives(
            which=which, default=self, alternatives=kwargs, default_key=default_key
        )


class RunnableSequence(RunnableSerializable[Input, Output]):
    """A sequence of runnables, where the output of each is the input of the next.

    RunnableSequence is the most important composition operator in LangChain as it is
    used in virtually every chain.

    A RunnableSequence can be instantiated directly or more commonly by using the `|`
    operator where either the left or right operands (or both) must be a Runnable.

    Any RunnableSequence automatically supports sync, async, batch.

    The default implementations of `batch` and `abatch` utilize threadpools and
    asyncio gather and will be faster than naive invocation of invoke or ainvoke
    for IO bound runnables.

    Batching is implemented by invoking the batch method on each component of the
    RunnableSequence in order.

    A RunnableSequence preserves the streaming properties of its components, so if all
    components of the sequence implement a `transform` method -- which
    is the method that implements the logic to map a streaming input to a streaming
    output -- then the sequence will be able to stream input to output!

    If any component of the sequence does not implement transform then the
    streaming will only begin after this component is run. If there are
    multiple blocking components, streaming begins after the last one.

    Please note: RunnableLambdas do not support `transform` by default! So if
        you need to use a RunnableLambdas be careful about where you place them in a
        RunnableSequence (if you need to use the .stream()/.astream() methods).

        If you need arbitrary logic and need streaming, you can subclass
        Runnable, and implement `transform` for whatever logic you need.

    Here is a simple example that uses simple functions to illustrate the use of
    RunnableSequence:

        .. code-block:: python

            from langchain.schema.runnable import RunnableLambda

            def add_one(x: int) -> int:
                return x + 1

            def mul_two(x: int) -> int:
                return x * 2

            runnable_1 = RunnableLambda(add_one)
            runnable_2 = RunnableLambda(mul_two)
            sequence = runnable_1 | runnable_2
            # Or equivalently:
            # sequence = RunnableSequence(first=runnable_1, last=runnable_2)
            sequence.invoke(1)
            await runnable.ainvoke(1)

            sequence.batch([1, 2, 3])
            await sequence.abatch([1, 2, 3])

    Here's an example that uses streams JSON output generated by an LLM:

        .. code-block:: python

            from langchain.output_parsers.json import SimpleJsonOutputParser
            from langchain.chat_models.openai import ChatOpenAI

            prompt = PromptTemplate.from_template(
                'In JSON format, give me a list of {topic} and their '
                'corresponding names in French, Spanish and in a '
                'Cat Language.'
            )

            model = ChatOpenAI()
            chain = prompt | model | SimpleJsonOutputParser()

            async for chunk in chain.astream({'topic': 'colors'}):
                print('-')
                print(chunk, sep='', flush=True)
    """

    # The steps are broken into first, middle and last, solely for type checking
    # purposes. It allows specifying the `Input` on the first type, the `Output` of
    # the last type.
    first: Runnable[Input, Any]
    """The first runnable in the sequence."""
    middle: List[Runnable[Any, Any]] = Field(default_factory=list)
    """The middle runnables in the sequence."""
    last: Runnable[Any, Output]
    """The last runnable in the sequence."""

    @property
    def steps(self) -> List[Runnable[Any, Any]]:
        """All the runnables that make up the sequence in order."""
        return [self.first] + self.middle + [self.last]

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        return cls.__module__.split(".")[:-1]

    class Config:
        arbitrary_types_allowed = True

    @property
    def InputType(self) -> Type[Input]:
        return self.first.InputType

    @property
    def OutputType(self) -> Type[Output]:
        return self.last.OutputType

    def get_input_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> Type[BaseModel]:
        from langchain.schema.runnable.passthrough import RunnableAssign

        if isinstance(self.first, RunnableAssign):
            first = cast(RunnableAssign, self.first)
            next_ = self.middle[0] if self.middle else self.last
            next_input_schema = next_.get_input_schema(config)
            if not next_input_schema.__custom_root_type__:
                # it's a dict as expected
                return create_model(  # type: ignore[call-overload]
                    "RunnableSequenceInput",
                    **{
                        k: (v.annotation, v.default)
                        for k, v in next_input_schema.__fields__.items()
                        if k not in first.mapper.steps
                    },
                )

        return self.first.get_input_schema(config)

    def get_output_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> Type[BaseModel]:
        return self.last.get_output_schema(config)

    @property
    def config_specs(self) -> List[ConfigurableFieldSpec]:
        return get_unique_config_specs(
            spec for step in self.steps for spec in step.config_specs
        )

    def __repr__(self) -> str:
        return "\n| ".join(
            repr(s) if i == 0 else indent_lines_after_first(repr(s), "| ")
            for i, s in enumerate(self.steps)
        )

    def __or__(
        self,
        other: Union[
            Runnable[Any, Other],
            Callable[[Any], Other],
            Callable[[Iterator[Any]], Iterator[Other]],
            Mapping[str, Union[Runnable[Any, Other], Callable[[Any], Other], Any]],
        ],
    ) -> RunnableSerializable[Input, Other]:
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
            Callable[[Other], Any],
            Callable[[Iterator[Other]], Iterator[Any]],
            Mapping[str, Union[Runnable[Other, Any], Callable[[Other], Any], Any]],
        ],
    ) -> RunnableSerializable[Other, Output]:
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
        except BaseException as e:
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
        except BaseException as e:
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

        if not inputs:
            return []

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
        except BaseException as e:
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

        if not inputs:
            return []

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
        except BaseException as e:
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

    def _transform(
        self,
        input: Iterator[Input],
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
    ) -> Iterator[Output]:
        steps = [self.first] + self.middle + [self.last]

        # transform the input stream of each step with the next
        # steps that don't natively support transforming an input stream will
        # buffer input in memory until all available, and then start emitting output
        final_pipeline = cast(Iterator[Output], input)
        for step in steps:
            final_pipeline = step.transform(
                final_pipeline,
                patch_config(
                    config,
                    callbacks=run_manager.get_child(f"seq:step:{steps.index(step)+1}"),
                ),
            )

        for output in final_pipeline:
            yield output

    async def _atransform(
        self,
        input: AsyncIterator[Input],
        run_manager: AsyncCallbackManagerForChainRun,
        config: RunnableConfig,
    ) -> AsyncIterator[Output]:
        steps = [self.first] + self.middle + [self.last]

        # stream the last steps
        # transform the input stream of each step with the next
        # steps that don't natively support transforming an input stream will
        # buffer input in memory until all available, and then start emitting output
        final_pipeline = cast(AsyncIterator[Output], input)
        for step in steps:
            final_pipeline = step.atransform(
                final_pipeline,
                patch_config(
                    config,
                    callbacks=run_manager.get_child(f"seq:step:{steps.index(step)+1}"),
                ),
            )
        async for output in final_pipeline:
            yield output

    def transform(
        self,
        input: Iterator[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Output]:
        yield from self._transform_stream_with_config(
            input, self._transform, config, **kwargs
        )

    def stream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Output]:
        yield from self.transform(iter([input]), config, **kwargs)

    async def atransform(
        self,
        input: AsyncIterator[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Output]:
        async for chunk in self._atransform_stream_with_config(
            input, self._atransform, config, **kwargs
        ):
            yield chunk

    async def astream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Output]:
        async def input_aiter() -> AsyncIterator[Input]:
            yield input

        async for chunk in self.atransform(input_aiter(), config, **kwargs):
            yield chunk


class RunnableParallel(RunnableSerializable[Input, Dict[str, Any]]):
    """
    A runnable that runs a mapping of runnables in parallel,
    and returns a mapping of their outputs.
    """

    steps: Mapping[str, Runnable[Input, Any]]

    def __init__(
        self,
        __steps: Optional[
            Mapping[
                str,
                Union[
                    Runnable[Input, Any],
                    Callable[[Input], Any],
                    Mapping[str, Union[Runnable[Input, Any], Callable[[Input], Any]]],
                ],
            ]
        ] = None,
        **kwargs: Union[
            Runnable[Input, Any],
            Callable[[Input], Any],
            Mapping[str, Union[Runnable[Input, Any], Callable[[Input], Any]]],
        ],
    ) -> None:
        merged = {**__steps} if __steps is not None else {}
        merged.update(kwargs)
        super().__init__(
            steps={key: coerce_to_runnable(r) for key, r in merged.items()}
        )

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        return cls.__module__.split(".")[:-1]

    class Config:
        arbitrary_types_allowed = True

    @property
    def InputType(self) -> Any:
        for step in self.steps.values():
            if step.InputType:
                return step.InputType

        return Any

    def get_input_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> Type[BaseModel]:
        if all(
            s.get_input_schema(config).schema().get("type", "object") == "object"
            for s in self.steps.values()
        ):
            # This is correct, but pydantic typings/mypy don't think so.
            return create_model(  # type: ignore[call-overload]
                "RunnableParallelInput",
                **{
                    k: (v.annotation, v.default)
                    for step in self.steps.values()
                    for k, v in step.get_input_schema(config).__fields__.items()
                    if k != "__root__"
                },
            )

        return super().get_input_schema(config)

    def get_output_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> Type[BaseModel]:
        # This is correct, but pydantic typings/mypy don't think so.
        return create_model(  # type: ignore[call-overload]
            "RunnableParallelOutput",
            **{k: (v.OutputType, None) for k, v in self.steps.items()},
        )

    @property
    def config_specs(self) -> List[ConfigurableFieldSpec]:
        return get_unique_config_specs(
            spec for step in self.steps.values() for spec in step.config_specs
        )

    def __repr__(self) -> str:
        map_for_repr = ",\n  ".join(
            f"{k}: {indent_lines_after_first(repr(v), '  ' + k + ': ')}"
            for k, v in self.steps.items()
        )
        return "{\n  " + map_for_repr + "\n}"

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
                            callbacks=run_manager.get_child(f"map:key:{key}"),
                        ),
                    )
                    for key, step in steps.items()
                ]
                output = {key: future.result() for key, future in zip(steps, futures)}
        # finish the root run
        except BaseException as e:
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
        except BaseException as e:
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
    ) -> Iterator[AddableDict]:
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
                        chunk = AddableDict({step_name: future.result()})
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
    ) -> AsyncIterator[AddableDict]:
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
                    chunk = AddableDict({step_name: task.result()})
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


# We support both names
RunnableMap = RunnableParallel


class RunnableGenerator(Runnable[Input, Output]):
    """
    A runnable that runs a generator function.
    """

    def __init__(
        self,
        transform: Union[
            Callable[[Iterator[Input]], Iterator[Output]],
            Callable[[AsyncIterator[Input]], AsyncIterator[Output]],
        ],
        atransform: Optional[
            Callable[[AsyncIterator[Input]], AsyncIterator[Output]]
        ] = None,
    ) -> None:
        if atransform is not None:
            self._atransform = atransform

        if inspect.isasyncgenfunction(transform):
            self._atransform = transform
        elif inspect.isgeneratorfunction(transform):
            self._transform = transform
        else:
            raise TypeError(
                "Expected a generator function type for `transform`."
                f"Instead got an unsupported type: {type(transform)}"
            )

    @property
    def InputType(self) -> Any:
        func = getattr(self, "_transform", None) or getattr(self, "_atransform")
        try:
            params = inspect.signature(func).parameters
            first_param = next(iter(params.values()), None)
            if first_param and first_param.annotation != inspect.Parameter.empty:
                return getattr(first_param.annotation, "__args__", (Any,))[0]
            else:
                return Any
        except ValueError:
            return Any

    @property
    def OutputType(self) -> Any:
        func = getattr(self, "_transform", None) or getattr(self, "_atransform")
        try:
            sig = inspect.signature(func)
            return (
                getattr(sig.return_annotation, "__args__", (Any,))[0]
                if sig.return_annotation != inspect.Signature.empty
                else Any
            )
        except ValueError:
            return Any

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, RunnableGenerator):
            if hasattr(self, "_transform") and hasattr(other, "_transform"):
                return self._transform == other._transform
            elif hasattr(self, "_atransform") and hasattr(other, "_atransform"):
                return self._atransform == other._atransform
            else:
                return False
        else:
            return False

    def __repr__(self) -> str:
        return "RunnableGenerator(...)"

    def transform(
        self,
        input: Iterator[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Iterator[Output]:
        return self._transform_stream_with_config(
            input, self._transform, config, **kwargs
        )

    def stream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Iterator[Output]:
        return self.transform(iter([input]), config, **kwargs)

    def invoke(
        self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Output:
        final = None
        for output in self.stream(input, config, **kwargs):
            if final is None:
                final = output
            else:
                final = final + output
        return cast(Output, final)

    def atransform(
        self,
        input: AsyncIterator[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Output]:
        if not hasattr(self, "_atransform"):
            raise NotImplementedError("This runnable does not support async methods.")

        return self._atransform_stream_with_config(
            input, self._atransform, config, **kwargs
        )

    def astream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Output]:
        async def input_aiter() -> AsyncIterator[Input]:
            yield input

        return self.atransform(input_aiter(), config, **kwargs)

    async def ainvoke(
        self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Output:
        final = None
        async for output in self.astream(input, config, **kwargs):
            if final is None:
                final = output
            else:
                final = final + output
        return cast(Output, final)


class RunnableLambda(Runnable[Input, Output]):
    """RunnableLambda converts a python callable into a Runnable.

    Wrapping a callable in a RunnableLambda makes the callable usable
    within either a sync or async context.

    RunnableLambda can be composed as any other Runnable and provides
    seamless integration with LangChain tracing.

    Examples:

        .. code-block:: python

            # This is a RunnableLambda
            from langchain.schema.runnable import RunnableLambda

            def add_one(x: int) -> int:
                return x + 1

            runnable = RunnableLambda(add_one)

            runnable.invoke(1) # returns 2
            runnable.batch([1, 2, 3]) # returns [2, 3, 4]

            # Async is supported by default by delegating to the sync implementation
            await runnable.ainvoke(1) # returns 2
            await runnable.abatch([1, 2, 3]) # returns [2, 3, 4]


            # Alternatively, can provide both synd and sync implementations
            async def add_one_async(x: int) -> int:
                return x + 1

            runnable = RunnableLambda(add_one, afunc=add_one_async)
            runnable.invoke(1) # Uses add_one
            await runnable.ainvoke(1) # Uses add_one_async
    """

    def __init__(
        self,
        func: Union[
            Union[
                Callable[[Input], Output],
                Callable[[Input, RunnableConfig], Output],
                Callable[[Input, CallbackManagerForChainRun], Output],
                Callable[[Input, CallbackManagerForChainRun, RunnableConfig], Output],
            ],
            Union[
                Callable[[Input], Awaitable[Output]],
                Callable[[Input, RunnableConfig], Awaitable[Output]],
                Callable[[Input, AsyncCallbackManagerForChainRun], Awaitable[Output]],
                Callable[
                    [Input, AsyncCallbackManagerForChainRun, RunnableConfig],
                    Awaitable[Output],
                ],
            ],
        ],
        afunc: Optional[
            Union[
                Callable[[Input], Awaitable[Output]],
                Callable[[Input, RunnableConfig], Awaitable[Output]],
                Callable[[Input, AsyncCallbackManagerForChainRun], Awaitable[Output]],
                Callable[
                    [Input, AsyncCallbackManagerForChainRun, RunnableConfig],
                    Awaitable[Output],
                ],
            ]
        ] = None,
    ) -> None:
        """Create a RunnableLambda from a callable, and async callable or both.

        Accepts both sync and async variants to allow providing efficient
        implementations for sync and async execution.

        Args:
            func: Either sync or async callable
            afunc: An async callable that takes an input and returns an output.
        """
        if afunc is not None:
            self.afunc = afunc

        if inspect.iscoroutinefunction(func):
            if afunc is not None:
                raise TypeError(
                    "Func was provided as a coroutine function, but afunc was "
                    "also provided. If providing both, func should be a regular "
                    "function to avoid ambiguity."
                )
            self.afunc = func
        elif callable(func):
            self.func = cast(Callable[[Input], Output], func)
        else:
            raise TypeError(
                "Expected a callable type for `func`."
                f"Instead got an unsupported type: {type(func)}"
            )

    @property
    def InputType(self) -> Any:
        """The type of the input to this runnable."""
        func = getattr(self, "func", None) or getattr(self, "afunc")
        try:
            params = inspect.signature(func).parameters
            first_param = next(iter(params.values()), None)
            if first_param and first_param.annotation != inspect.Parameter.empty:
                return first_param.annotation
            else:
                return Any
        except ValueError:
            return Any

    def get_input_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> Type[BaseModel]:
        """The pydantic schema for the input to this runnable."""
        func = getattr(self, "func", None) or getattr(self, "afunc")

        if isinstance(func, itemgetter):
            # This is terrible, but afaict it's not possible to access _items
            # on itemgetter objects, so we have to parse the repr
            items = str(func).replace("operator.itemgetter(", "")[:-1].split(", ")
            if all(
                item[0] == "'" and item[-1] == "'" and len(item) > 2 for item in items
            ):
                # It's a dict, lol
                return create_model(
                    "RunnableLambdaInput",
                    **{item[1:-1]: (Any, None) for item in items},  # type: ignore
                )
            else:
                return create_model("RunnableLambdaInput", __root__=(List[Any], None))

        if self.InputType != Any:
            return super().get_input_schema(config)

        if dict_keys := get_function_first_arg_dict_keys(func):
            return create_model(
                "RunnableLambdaInput",
                **{key: (Any, None) for key in dict_keys},  # type: ignore
            )

        return super().get_input_schema(config)

    @property
    def OutputType(self) -> Any:
        """The type of the output of this runnable as a type annotation."""
        func = getattr(self, "func", None) or getattr(self, "afunc")
        try:
            sig = inspect.signature(func)
            return (
                sig.return_annotation
                if sig.return_annotation != inspect.Signature.empty
                else Any
            )
        except ValueError:
            return Any

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
        """A string representation of this runnable."""
        if hasattr(self, "func"):
            return f"RunnableLambda({get_lambda_source(self.func) or '...'})"
        elif hasattr(self, "afunc"):
            return f"RunnableLambda(afunc={get_lambda_source(self.afunc) or '...'})"
        else:
            return "RunnableLambda(...)"

    def _invoke(
        self,
        input: Input,
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> Output:
        output = call_func_with_variable_args(
            self.func, input, config, run_manager, **kwargs
        )
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
        **kwargs: Any,
    ) -> Output:
        output = await acall_func_with_variable_args(
            self.afunc, input, config, run_manager, **kwargs
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
        """Invoke this runnable synchronously."""
        if hasattr(self, "func"):
            return self._call_with_config(
                self._invoke,
                input,
                self._config(config, self.func),
                **kwargs,
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
        """Invoke this runnable asynchronously."""
        if hasattr(self, "afunc"):
            return await self._acall_with_config(
                self._ainvoke,
                input,
                self._config(config, self.afunc),
                **kwargs,
            )
        else:
            # Delegating to super implementation of ainvoke.
            # Uses asyncio executor to run the sync version (invoke)
            return await super().ainvoke(input, config)


class RunnableEachBase(RunnableSerializable[List[Input], List[Output]]):
    """
    A runnable that delegates calls to another runnable
    with each element of the input sequence.

    Use only if creating a new RunnableEach subclass with different __init__ args.
    """

    bound: Runnable[Input, Output]

    class Config:
        arbitrary_types_allowed = True

    @property
    def InputType(self) -> Any:
        return List[self.bound.InputType]  # type: ignore[name-defined]

    def get_input_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> Type[BaseModel]:
        return create_model(
            "RunnableEachInput",
            __root__=(
                List[self.bound.get_input_schema(config)],  # type: ignore
                None,
            ),
        )

    @property
    def OutputType(self) -> Type[List[Output]]:
        return List[self.bound.OutputType]  # type: ignore[name-defined]

    def get_output_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> Type[BaseModel]:
        schema = self.bound.get_output_schema(config)
        return create_model(
            "RunnableEachOutput",
            __root__=(
                List[schema],  # type: ignore
                None,
            ),
        )

    @property
    def config_specs(self) -> List[ConfigurableFieldSpec]:
        return self.bound.config_specs

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        return cls.__module__.split(".")[:-1]

    def _invoke(
        self,
        inputs: List[Input],
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> List[Output]:
        return self.bound.batch(
            inputs, patch_config(config, callbacks=run_manager.get_child()), **kwargs
        )

    def invoke(
        self, input: List[Input], config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> List[Output]:
        return self._call_with_config(self._invoke, input, config, **kwargs)

    async def _ainvoke(
        self,
        inputs: List[Input],
        run_manager: AsyncCallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> List[Output]:
        return await self.bound.abatch(
            inputs, patch_config(config, callbacks=run_manager.get_child()), **kwargs
        )

    async def ainvoke(
        self, input: List[Input], config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> List[Output]:
        return await self._acall_with_config(self._ainvoke, input, config, **kwargs)


class RunnableEach(RunnableEachBase[Input, Output]):
    """
    A runnable that delegates calls to another runnable
    with each element of the input sequence.
    """

    def bind(self, **kwargs: Any) -> RunnableEach[Input, Output]:
        return RunnableEach(bound=self.bound.bind(**kwargs))

    def with_config(
        self, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> RunnableEach[Input, Output]:
        return RunnableEach(bound=self.bound.with_config(config, **kwargs))

    def with_listeners(
        self,
        *,
        on_start: Optional[Listener] = None,
        on_end: Optional[Listener] = None,
        on_error: Optional[Listener] = None,
    ) -> RunnableEach[Input, Output]:
        """
        Bind lifecycle listeners to a Runnable, returning a new Runnable.

        on_start: Called before the runnable starts running, with the Run object.
        on_end: Called after the runnable finishes running, with the Run object.
        on_error: Called if the runnable throws an error, with the Run object.

        The Run object contains information about the run, including its id,
        type, input, output, error, start_time, end_time, and any tags or metadata
        added to the run.
        """
        return RunnableEach(
            bound=self.bound.with_listeners(
                on_start=on_start, on_end=on_end, on_error=on_error
            )
        )


class RunnableBindingBase(RunnableSerializable[Input, Output]):
    """
    A runnable that delegates calls to another runnable with a set of kwargs.

    Use only if creating a new RunnableBinding subclass with different __init__ args.
    """

    bound: Runnable[Input, Output]

    kwargs: Mapping[str, Any] = Field(default_factory=dict)

    config: RunnableConfig = Field(default_factory=dict)

    config_factories: List[Callable[[RunnableConfig], RunnableConfig]] = Field(
        default_factory=list
    )

    # Union[Type[Input], BaseModel] + things like List[str]
    custom_input_type: Optional[Any] = None
    # Union[Type[Output], BaseModel] + things like List[str]
    custom_output_type: Optional[Any] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        *,
        bound: Runnable[Input, Output],
        kwargs: Optional[Mapping[str, Any]] = None,
        config: Optional[RunnableConfig] = None,
        config_factories: Optional[
            List[Callable[[RunnableConfig], RunnableConfig]]
        ] = None,
        custom_input_type: Optional[Union[Type[Input], BaseModel]] = None,
        custom_output_type: Optional[Union[Type[Output], BaseModel]] = None,
        **other_kwargs: Any,
    ) -> None:
        config = config or {}
        # config_specs contains the list of valid `configurable` keys
        if configurable := config.get("configurable", None):
            allowed_keys = set(s.id for s in bound.config_specs)
            for key in configurable:
                if key not in allowed_keys:
                    raise ValueError(
                        f"Configurable key '{key}' not found in runnable with"
                        f" config keys: {allowed_keys}"
                    )
        super().__init__(
            bound=bound,
            kwargs=kwargs or {},
            config=config or {},
            config_factories=config_factories or [],
            custom_input_type=custom_input_type,
            custom_output_type=custom_output_type,
            **other_kwargs,
        )

    @property
    def InputType(self) -> Type[Input]:
        return (
            cast(Type[Input], self.custom_input_type)
            if self.custom_input_type is not None
            else self.bound.InputType
        )

    @property
    def OutputType(self) -> Type[Output]:
        return (
            cast(Type[Output], self.custom_output_type)
            if self.custom_output_type is not None
            else self.bound.OutputType
        )

    def get_input_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> Type[BaseModel]:
        if self.custom_input_type is not None:
            return super().get_input_schema(config)
        return self.bound.get_input_schema(merge_configs(self.config, config))

    def get_output_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> Type[BaseModel]:
        if self.custom_output_type is not None:
            return super().get_output_schema(config)
        return self.bound.get_output_schema(merge_configs(self.config, config))

    @property
    def config_specs(self) -> List[ConfigurableFieldSpec]:
        return self.bound.config_specs

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        return cls.__module__.split(".")[:-1]

    def _merge_configs(self, *configs: Optional[RunnableConfig]) -> RunnableConfig:
        config = merge_configs(self.config, *configs)
        return merge_configs(config, *(f(config) for f in self.config_factories))

    def invoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Output:
        return self.bound.invoke(
            input,
            self._merge_configs(config),
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
            self._merge_configs(config),
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
                List[RunnableConfig],
                [self._merge_configs(conf) for conf in config],
            )
        else:
            configs = [self._merge_configs(config) for _ in range(len(inputs))]
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
                List[RunnableConfig],
                [self._merge_configs(conf) for conf in config],
            )
        else:
            configs = [self._merge_configs(config) for _ in range(len(inputs))]
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
            self._merge_configs(config),
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
            self._merge_configs(config),
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
            self._merge_configs(config),
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
            self._merge_configs(config),
            **{**self.kwargs, **kwargs},
        ):
            yield item


RunnableBindingBase.update_forward_refs(RunnableConfig=RunnableConfig)


class RunnableBinding(RunnableBindingBase[Input, Output]):
    """
    A runnable that delegates calls to another runnable with a set of kwargs.
    """

    def bind(self, **kwargs: Any) -> Runnable[Input, Output]:
        return self.__class__(
            bound=self.bound,
            config=self.config,
            kwargs={**self.kwargs, **kwargs},
            custom_input_type=self.custom_input_type,
            custom_output_type=self.custom_output_type,
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
            config=cast(RunnableConfig, {**self.config, **(config or {}), **kwargs}),
            custom_input_type=self.custom_input_type,
            custom_output_type=self.custom_output_type,
        )

    def with_listeners(
        self,
        *,
        on_start: Optional[Listener] = None,
        on_end: Optional[Listener] = None,
        on_error: Optional[Listener] = None,
    ) -> Runnable[Input, Output]:
        """
        Bind lifecycle listeners to a Runnable, returning a new Runnable.

        on_start: Called before the runnable starts running, with the Run object.
        on_end: Called after the runnable finishes running, with the Run object.
        on_error: Called if the runnable throws an error, with the Run object.

        The Run object contains information about the run, including its id,
        type, input, output, error, start_time, end_time, and any tags or metadata
        added to the run.
        """
        from langchain.callbacks.tracers.root_listeners import RootListenersTracer

        return self.__class__(
            bound=self.bound,
            kwargs=self.kwargs,
            config=self.config,
            config_factories=[
                lambda config: {
                    "callbacks": [
                        RootListenersTracer(
                            config=config,
                            on_start=on_start,
                            on_end=on_end,
                            on_error=on_error,
                        )
                    ],
                }
            ],
            custom_input_type=self.custom_input_type,
            custom_output_type=self.custom_output_type,
        )

    def with_types(
        self,
        input_type: Optional[Union[Type[Input], BaseModel]] = None,
        output_type: Optional[Union[Type[Output], BaseModel]] = None,
    ) -> Runnable[Input, Output]:
        return self.__class__(
            bound=self.bound,
            kwargs=self.kwargs,
            config=self.config,
            custom_input_type=input_type
            if input_type is not None
            else self.custom_input_type,
            custom_output_type=output_type
            if output_type is not None
            else self.custom_output_type,
        )

    def with_retry(self, **kwargs: Any) -> Runnable[Input, Output]:
        return self.__class__(
            bound=self.bound.with_retry(**kwargs),
            kwargs=self.kwargs,
            config=self.config,
        )


RunnableLike = Union[
    Runnable[Input, Output],
    Callable[[Input], Output],
    Callable[[Input], Awaitable[Output]],
    Callable[[Iterator[Input]], Iterator[Output]],
    Callable[[AsyncIterator[Input]], AsyncIterator[Output]],
    Mapping[str, Any],
]


def coerce_to_runnable(thing: RunnableLike) -> Runnable[Input, Output]:
    """Coerce a runnable-like object into a Runnable.

    Args:
        thing: A runnable-like object.

    Returns:
        A Runnable.
    """
    if isinstance(thing, Runnable):
        return thing
    elif inspect.isasyncgenfunction(thing) or inspect.isgeneratorfunction(thing):
        return RunnableGenerator(thing)
    elif callable(thing):
        return RunnableLambda(cast(Callable[[Input], Output], thing))
    elif isinstance(thing, dict):
        return cast(Runnable[Input, Output], RunnableParallel(thing))
    else:
        raise TypeError(
            f"Expected a Runnable, callable or dict."
            f"Instead got an unsupported type: {type(thing)}"
        )
