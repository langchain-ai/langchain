from __future__ import annotations

from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Union,
    cast,
)

from typing_extensions import TypedDict

from langchain.schema.runnable.utils import (
    Input,
    Output,
    accepts_config,
    accepts_run_manager,
)

if TYPE_CHECKING:
    from langchain.callbacks.base import BaseCallbackManager, Callbacks
    from langchain.callbacks.manager import (
        AsyncCallbackManager,
        AsyncCallbackManagerForChainRun,
        CallbackManager,
        CallbackManagerForChainRun,
    )


class EmptyDict(TypedDict, total=False):
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

    locals: Dict[str, Any]
    """
    Variables scoped to this call and any sub-calls. Usually used with
    GetLocalVar() and PutLocalVar(). Care should be taken when placing mutable
    objects in locals, as they will be shared between parallel sub-calls.
    """

    max_concurrency: Optional[int]
    """
    Maximum number of parallel calls to make. If not provided, defaults to 
    ThreadPoolExecutor's default. This is ignored if an executor is provided.
    """

    recursion_limit: int
    """
    Maximum number of times a call can recurse. If not provided, defaults to 10.
    """

    configurable: Dict[str, Any]
    """
    Runtime values for attributes previously made configurable by this Runnable,
    or sub-Runnables, through .make_configurable(). Check .output_schema for
    a description of the attributes that have been made configurable.
    """


def ensure_config(config: Optional[RunnableConfig] = None) -> RunnableConfig:
    empty = RunnableConfig(
        tags=[],
        metadata={},
        callbacks=None,
        locals={},
        recursion_limit=10,
    )
    if config is not None:
        empty.update(
            cast(RunnableConfig, {k: v for k, v in config.items() if v is not None})
        )
    return empty


def get_config_list(
    config: Optional[Union[RunnableConfig, List[RunnableConfig]]], length: int
) -> List[RunnableConfig]:
    """
    Helper method to get a list of configs from a single config or a list of
    configs, useful for subclasses overriding batch() or abatch().
    """
    if length < 0:
        raise ValueError(f"length must be >= 0, but got {length}")
    if isinstance(config, list) and len(config) != length:
        raise ValueError(
            f"config must be a list of the same length as inputs, "
            f"but got {len(config)} configs for {length} inputs"
        )

    return (
        list(map(ensure_config, config))
        if isinstance(config, list)
        else [patch_config(config, copy_locals=True) for _ in range(length)]
    )


def patch_config(
    config: Optional[RunnableConfig],
    *,
    copy_locals: bool = False,
    callbacks: Optional[BaseCallbackManager] = None,
    recursion_limit: Optional[int] = None,
    max_concurrency: Optional[int] = None,
    run_name: Optional[str] = None,
    configurable: Optional[Dict[str, Any]] = None,
) -> RunnableConfig:
    config = ensure_config(config)
    if copy_locals:
        config["locals"] = config["locals"].copy()
    if callbacks is not None:
        # If we're replacing callbacks we need to unset run_name
        # As that should apply only to the same run as the original callbacks
        config["callbacks"] = callbacks
        if "run_name" in config:
            del config["run_name"]
    if recursion_limit is not None:
        config["recursion_limit"] = recursion_limit
    if max_concurrency is not None:
        config["max_concurrency"] = max_concurrency
    if run_name is not None:
        config["run_name"] = run_name
    if configurable is not None:
        config["configurable"] = {**config.get("configurable", {}), **configurable}
    return config


def call_func_with_variable_args(
    func: Union[
        Callable[[Input], Output],
        Callable[[Input, CallbackManagerForChainRun], Output],
        Callable[[Input, CallbackManagerForChainRun, RunnableConfig], Output],
    ],
    input: Input,
    run_manager: CallbackManagerForChainRun,
    config: RunnableConfig,
    **kwargs: Any,
) -> Output:
    """Call function that may optionally accept a run_manager and/or config."""
    if accepts_config(func):
        kwargs["config"] = patch_config(config, callbacks=run_manager.get_child())
    if accepts_run_manager(func):
        kwargs["run_manager"] = run_manager
    return func(input, **kwargs)  # type: ignore[call-arg]


async def acall_func_with_variable_args(
    func: Union[
        Callable[[Input], Awaitable[Output]],
        Callable[[Input, AsyncCallbackManagerForChainRun], Awaitable[Output]],
        Callable[
            [Input, AsyncCallbackManagerForChainRun, RunnableConfig],
            Awaitable[Output],
        ],
    ],
    input: Input,
    run_manager: AsyncCallbackManagerForChainRun,
    config: RunnableConfig,
    **kwargs: Any,
) -> Output:
    """Call function that may optionally accept a run_manager and/or config."""
    if accepts_config(func):
        kwargs["config"] = patch_config(config, callbacks=run_manager.get_child())
    if accepts_run_manager(func):
        kwargs["run_manager"] = run_manager
    return await func(input, **kwargs)  # type: ignore[call-arg]


def get_callback_manager_for_config(config: RunnableConfig) -> CallbackManager:
    from langchain.callbacks.manager import CallbackManager

    return CallbackManager.configure(
        inheritable_callbacks=config.get("callbacks"),
        inheritable_tags=config.get("tags"),
        inheritable_metadata=config.get("metadata"),
    )


def get_async_callback_manager_for_config(
    config: RunnableConfig,
) -> AsyncCallbackManager:
    from langchain.callbacks.manager import AsyncCallbackManager

    return AsyncCallbackManager.configure(
        inheritable_callbacks=config.get("callbacks"),
        inheritable_tags=config.get("tags"),
        inheritable_metadata=config.get("metadata"),
    )


@contextmanager
def get_executor_for_config(config: RunnableConfig) -> Generator[Executor, None, None]:
    with ThreadPoolExecutor(max_workers=config.get("max_concurrency")) as executor:
        yield executor
