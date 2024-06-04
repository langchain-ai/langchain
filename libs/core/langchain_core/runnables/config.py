from __future__ import annotations

import asyncio
import uuid
import warnings
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
    cast,
)

from typing_extensions import ParamSpec, TypedDict

from langchain_core.runnables.utils import (
    Input,
    Output,
    accepts_config,
    accepts_run_manager,
)

if TYPE_CHECKING:
    from langchain_core.callbacks.base import BaseCallbackManager, Callbacks
    from langchain_core.callbacks.manager import (
        AsyncCallbackManager,
        AsyncCallbackManagerForChainRun,
        CallbackManager,
        CallbackManagerForChainRun,
    )
else:
    # Pydantic validates through typed dicts, but
    # the callbacks need forward refs updated
    Callbacks = Optional[Union[List, Any]]


class EmptyDict(TypedDict, total=False):
    """Empty dict type."""

    pass


class RunnableConfig(TypedDict, total=False):
    """Configuration for a Runnable."""

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

    run_name: str
    """
    Name for the tracer run for this call. Defaults to the name of the class.
    """

    max_concurrency: Optional[int]
    """
    Maximum number of parallel calls to make. If not provided, defaults to 
    ThreadPoolExecutor's default.
    """

    recursion_limit: int
    """
    Maximum number of times a call can recurse. If not provided, defaults to 25.
    """

    configurable: Dict[str, Any]
    """
    Runtime values for attributes previously made configurable on this Runnable,
    or sub-Runnables, through .configurable_fields() or .configurable_alternatives().
    Check .output_schema() for a description of the attributes that have been made 
    configurable.
    """

    run_id: Optional[uuid.UUID]
    """
    Unique identifier for the tracer run for this call. If not provided, a new UUID
        will be generated.
    """


var_child_runnable_config = ContextVar(
    "child_runnable_config", default=RunnableConfig()
)


def _set_config_context(config: RunnableConfig) -> None:
    """Set the child runnable config + tracing context

    Args:
        config (RunnableConfig): The config to set.
    """
    from langsmith import (
        RunTree,  # type: ignore
        run_helpers,  # type: ignore
    )

    var_child_runnable_config.set(config)
    if hasattr(RunTree, "from_runnable_config"):
        # import _set_tracing_context, get_tracing_context
        rt = RunTree.from_runnable_config(dict(config))
        tc = run_helpers.get_tracing_context()
        run_helpers._set_tracing_context({**tc, "parent": rt})


def ensure_config(config: Optional[RunnableConfig] = None) -> RunnableConfig:
    """Ensure that a config is a dict with all keys present.

    Args:
        config (Optional[RunnableConfig], optional): The config to ensure.
          Defaults to None.

    Returns:
        RunnableConfig: The ensured config.
    """
    empty = RunnableConfig(
        tags=[],
        metadata={},
        callbacks=None,
        recursion_limit=25,
    )
    if var_config := var_child_runnable_config.get():
        empty.update(
            cast(RunnableConfig, {k: v for k, v in var_config.items() if v is not None})
        )
    if config is not None:
        empty.update(
            cast(RunnableConfig, {k: v for k, v in config.items() if v is not None})
        )
    for key, value in empty.get("configurable", {}).items():
        if isinstance(value, (str, int, float, bool)) and key not in empty["metadata"]:
            empty["metadata"][key] = value
    return empty


def get_config_list(
    config: Optional[Union[RunnableConfig, Sequence[RunnableConfig]]], length: int
) -> List[RunnableConfig]:
    """Get a list of configs from a single config or a list of configs.

     It is useful for subclasses overriding batch() or abatch().

    Args:
        config (Optional[Union[RunnableConfig, List[RunnableConfig]]]):
          The config or list of configs.
        length (int): The length of the list.

    Returns:
        List[RunnableConfig]: The list of configs.

    Raises:
        ValueError: If the length of the list is not equal to the length of the inputs.

    """
    if length < 0:
        raise ValueError(f"length must be >= 0, but got {length}")
    if isinstance(config, Sequence) and len(config) != length:
        raise ValueError(
            f"config must be a list of the same length as inputs, "
            f"but got {len(config)} configs for {length} inputs"
        )

    if isinstance(config, Sequence):
        return list(map(ensure_config, config))
    if length > 1 and isinstance(config, dict) and config.get("run_id") is not None:
        warnings.warn(
            "Provided run_id be used only for the first element of the batch.",
            category=RuntimeWarning,
        )
        subsequent = cast(
            RunnableConfig, {k: v for k, v in config.items() if k != "run_id"}
        )
        return [
            ensure_config(subsequent) if i else ensure_config(config)
            for i in range(length)
        ]
    return [ensure_config(config) for i in range(length)]


def patch_config(
    config: Optional[RunnableConfig],
    *,
    callbacks: Optional[BaseCallbackManager] = None,
    recursion_limit: Optional[int] = None,
    max_concurrency: Optional[int] = None,
    run_name: Optional[str] = None,
    configurable: Optional[Dict[str, Any]] = None,
) -> RunnableConfig:
    """Patch a config with new values.

    Args:
        config (Optional[RunnableConfig]): The config to patch.
        copy_locals (bool, optional): Whether to copy locals. Defaults to False.
        callbacks (Optional[BaseCallbackManager], optional): The callbacks to set.
          Defaults to None.
        recursion_limit (Optional[int], optional): The recursion limit to set.
          Defaults to None.
        max_concurrency (Optional[int], optional): The max concurrency to set.
          Defaults to None.
        run_name (Optional[str], optional): The run name to set. Defaults to None.
        configurable (Optional[Dict[str, Any]], optional): The configurable to set.
          Defaults to None.

    Returns:
        RunnableConfig: The patched config.
    """
    config = ensure_config(config)
    if callbacks is not None:
        # If we're replacing callbacks, we need to unset run_name
        # As that should apply only to the same run as the original callbacks
        config["callbacks"] = callbacks
        if "run_name" in config:
            del config["run_name"]
        if "run_id" in config:
            del config["run_id"]
    if recursion_limit is not None:
        config["recursion_limit"] = recursion_limit
    if max_concurrency is not None:
        config["max_concurrency"] = max_concurrency
    if run_name is not None:
        config["run_name"] = run_name
    if configurable is not None:
        config["configurable"] = {**config.get("configurable", {}), **configurable}
    return config


def merge_configs(*configs: Optional[RunnableConfig]) -> RunnableConfig:
    """Merge multiple configs into one.

    Args:
        *configs (Optional[RunnableConfig]): The configs to merge.

    Returns:
        RunnableConfig: The merged config.
    """
    base: RunnableConfig = {}
    # Even though the keys aren't literals, this is correct
    # because both dicts are the same type
    for config in (c for c in configs if c is not None):
        for key in config:
            if key == "metadata":
                base[key] = {  # type: ignore
                    **base.get(key, {}),  # type: ignore
                    **(config.get(key) or {}),  # type: ignore
                }
            elif key == "tags":
                base[key] = list(  # type: ignore
                    set(base.get(key, []) + (config.get(key) or [])),  # type: ignore
                )
            elif key == "configurable":
                base[key] = {  # type: ignore
                    **base.get(key, {}),  # type: ignore
                    **(config.get(key) or {}),  # type: ignore
                }
            elif key == "callbacks":
                base_callbacks = base.get("callbacks")
                these_callbacks = config["callbacks"]
                # callbacks can be either None, list[handler] or manager
                # so merging two callbacks values has 6 cases
                if isinstance(these_callbacks, list):
                    if base_callbacks is None:
                        base["callbacks"] = these_callbacks
                    elif isinstance(base_callbacks, list):
                        base["callbacks"] = base_callbacks + these_callbacks
                    else:
                        # base_callbacks is a manager
                        mngr = base_callbacks.copy()
                        for callback in these_callbacks:
                            mngr.add_handler(callback, inherit=True)
                        base["callbacks"] = mngr
                elif these_callbacks is not None:
                    # these_callbacks is a manager
                    if base_callbacks is None:
                        base["callbacks"] = these_callbacks
                    elif isinstance(base_callbacks, list):
                        mngr = these_callbacks.copy()
                        for callback in base_callbacks:
                            mngr.add_handler(callback, inherit=True)
                        base["callbacks"] = mngr
                    else:
                        # base_callbacks is also a manager

                        manager = base_callbacks.__class__(
                            parent_run_id=base_callbacks.parent_run_id
                            or these_callbacks.parent_run_id,
                            handlers=[],
                            inheritable_handlers=[],
                            tags=list(set(base_callbacks.tags + these_callbacks.tags)),
                            inheritable_tags=list(
                                set(
                                    base_callbacks.inheritable_tags
                                    + these_callbacks.inheritable_tags
                                )
                            ),
                            metadata={
                                **base_callbacks.metadata,
                                **these_callbacks.metadata,
                            },
                        )

                        handlers = base_callbacks.handlers + these_callbacks.handlers
                        inheritable_handlers = (
                            base_callbacks.inheritable_handlers
                            + these_callbacks.inheritable_handlers
                        )

                        for handler in handlers:
                            manager.add_handler(handler)

                        for handler in inheritable_handlers:
                            manager.add_handler(handler, inherit=True)

                        base["callbacks"] = manager
            else:
                base[key] = config[key] or base.get(key)  # type: ignore
    return base


def call_func_with_variable_args(
    func: Union[
        Callable[[Input], Output],
        Callable[[Input, RunnableConfig], Output],
        Callable[[Input, CallbackManagerForChainRun], Output],
        Callable[[Input, CallbackManagerForChainRun, RunnableConfig], Output],
    ],
    input: Input,
    config: RunnableConfig,
    run_manager: Optional[CallbackManagerForChainRun] = None,
    **kwargs: Any,
) -> Output:
    """Call function that may optionally accept a run_manager and/or config.

    Args:
        func (Union[Callable[[Input], Output],
          Callable[[Input, CallbackManagerForChainRun], Output],
          Callable[[Input, CallbackManagerForChainRun, RunnableConfig], Output]]):
           The function to call.
        input (Input): The input to the function.
        run_manager (CallbackManagerForChainRun): The run manager to
          pass to the function.
        config (RunnableConfig): The config to pass to the function.
        **kwargs (Any): The keyword arguments to pass to the function.

    Returns:
        Output: The output of the function.
    """
    if accepts_config(func):
        if run_manager is not None:
            kwargs["config"] = patch_config(config, callbacks=run_manager.get_child())
        else:
            kwargs["config"] = config
    if run_manager is not None and accepts_run_manager(func):
        kwargs["run_manager"] = run_manager
    return func(input, **kwargs)  # type: ignore[call-arg]


def acall_func_with_variable_args(
    func: Union[
        Callable[[Input], Awaitable[Output]],
        Callable[[Input, RunnableConfig], Awaitable[Output]],
        Callable[[Input, AsyncCallbackManagerForChainRun], Awaitable[Output]],
        Callable[
            [Input, AsyncCallbackManagerForChainRun, RunnableConfig],
            Awaitable[Output],
        ],
    ],
    input: Input,
    config: RunnableConfig,
    run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    **kwargs: Any,
) -> Awaitable[Output]:
    """Call function that may optionally accept a run_manager and/or config.

    Args:
        func (Union[Callable[[Input], Awaitable[Output]], Callable[[Input,
            AsyncCallbackManagerForChainRun], Awaitable[Output]], Callable[[Input,
            AsyncCallbackManagerForChainRun, RunnableConfig], Awaitable[Output]]]):
            The function to call.
        input (Input): The input to the function.
        run_manager (AsyncCallbackManagerForChainRun): The run manager
          to pass to the function.
        config (RunnableConfig): The config to pass to the function.
        **kwargs (Any): The keyword arguments to pass to the function.

    Returns:
        Output: The output of the function.
    """
    if accepts_config(func):
        if run_manager is not None:
            kwargs["config"] = patch_config(config, callbacks=run_manager.get_child())
        else:
            kwargs["config"] = config
    if run_manager is not None and accepts_run_manager(func):
        kwargs["run_manager"] = run_manager
    return func(input, **kwargs)  # type: ignore[call-arg]


def get_callback_manager_for_config(config: RunnableConfig) -> CallbackManager:
    """Get a callback manager for a config.

    Args:
        config (RunnableConfig): The config.

    Returns:
        CallbackManager: The callback manager.
    """
    from langchain_core.callbacks.manager import CallbackManager

    return CallbackManager.configure(
        inheritable_callbacks=config.get("callbacks"),
        inheritable_tags=config.get("tags"),
        inheritable_metadata=config.get("metadata"),
    )


def get_async_callback_manager_for_config(
    config: RunnableConfig,
) -> AsyncCallbackManager:
    """Get an async callback manager for a config.

    Args:
        config (RunnableConfig): The config.

    Returns:
        AsyncCallbackManager: The async callback manager.
    """
    from langchain_core.callbacks.manager import AsyncCallbackManager

    return AsyncCallbackManager.configure(
        inheritable_callbacks=config.get("callbacks"),
        inheritable_tags=config.get("tags"),
        inheritable_metadata=config.get("metadata"),
    )


P = ParamSpec("P")
T = TypeVar("T")


class ContextThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor that copies the context to the child thread."""

    def submit(  # type: ignore[override]
        self,
        func: Callable[P, T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Future[T]:
        """Submit a function to the executor.

        Args:
            func (Callable[..., T]): The function to submit.
            *args (Any): The positional arguments to the function.
            **kwargs (Any): The keyword arguments to the function.

        Returns:
            Future[T]: The future for the function.
        """
        return super().submit(
            cast(Callable[..., T], partial(copy_context().run, func, *args, **kwargs))
        )

    def map(
        self,
        fn: Callable[..., T],
        *iterables: Iterable[Any],
        timeout: float | None = None,
        chunksize: int = 1,
    ) -> Iterator[T]:
        contexts = [copy_context() for _ in range(len(iterables[0]))]  # type: ignore[arg-type]

        def _wrapped_fn(*args: Any) -> T:
            return contexts.pop().run(fn, *args)

        return super().map(
            _wrapped_fn,
            *iterables,
            timeout=timeout,
            chunksize=chunksize,
        )


@contextmanager
def get_executor_for_config(
    config: Optional[RunnableConfig],
) -> Generator[Executor, None, None]:
    """Get an executor for a config.

    Args:
        config (RunnableConfig): The config.

    Yields:
        Generator[Executor, None, None]: The executor.
    """
    config = config or {}
    with ContextThreadPoolExecutor(
        max_workers=config.get("max_concurrency")
    ) as executor:
        yield executor


async def run_in_executor(
    executor_or_config: Optional[Union[Executor, RunnableConfig]],
    func: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    """Run a function in an executor.

    Args:
        executor (Executor): The executor.
        func (Callable[P, Output]): The function.
        *args (Any): The positional arguments to the function.
        **kwargs (Any): The keyword arguments to the function.

    Returns:
        Output: The output of the function.
    """
    if executor_or_config is None or isinstance(executor_or_config, dict):
        # Use default executor with context copied from current context
        return await asyncio.get_running_loop().run_in_executor(
            None,
            cast(Callable[..., T], partial(copy_context().run, func, *args, **kwargs)),
        )

    return await asyncio.get_running_loop().run_in_executor(
        executor_or_config, partial(func, **kwargs), *args
    )
