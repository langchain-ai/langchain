import asyncio
from itertools import tee
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    List,
    Optional,
    Union,
    cast,
)

from langchain.load.dump import dumpd
from langchain.schema.runnable.config import (
    RunnableConfig,
    acall_func_with_variable_args,
    call_func_with_variable_args,
    ensure_config,
    get_async_callback_manager_for_config,
    get_callback_manager_for_config,
    get_config_list,
    patch_config,
)
from langchain.schema.runnable.utils import (
    Input,
    Output,
    accepts_config,
    accepts_run_manager,
)
from langchain.utils.aiter import atee, py_anext

if TYPE_CHECKING:
    from langchain.callbacks.manager import (
        AsyncCallbackManagerForChainRun,
        CallbackManagerForChainRun,
    )


def call_with_config(
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
            func, input, run_manager, config, **kwargs
        )
    except BaseException as e:
        run_manager.on_chain_error(e)
        raise
    else:
        run_manager.on_chain_end(dumpd(output))
        return output


async def acall_with_config(
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
            func, input, run_manager, config, **kwargs
        )
    except BaseException as e:
        await run_manager.on_chain_error(e)
        raise
    else:
        await run_manager.on_chain_end(dumpd(output))
        return output


def batch_with_config(
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
        for callback_manager, input, config in zip(callback_managers, input, configs)
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


async def abatch_with_config(
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


def transform_stream_with_config(
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
            kwargs["config"] = patch_config(config, callbacks=run_manager.get_child())
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


async def atransform_stream_with_config(
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
            kwargs["config"] = patch_config(config, callbacks=run_manager.get_child())
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
