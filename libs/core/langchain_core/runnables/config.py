"""Configuration utilities for `Runnable` objects."""

from __future__ import annotations

import asyncio

# Cannot move uuid to TYPE_CHECKING as RunnableConfig is used in Pydantic models
import uuid  # noqa: TC003
import warnings
from collections.abc import Awaitable, Callable, Generator, Iterable, Iterator, Sequence
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import Context, ContextVar, Token, copy_context
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    ParamSpec,
    TypeVar,
    cast,
)

from langsmith.run_helpers import _set_tracing_context, get_tracing_context
from typing_extensions import TypedDict

from langchain_core.callbacks.manager import AsyncCallbackManager, CallbackManager
from langchain_core.runnables.utils import (
    Input,
    Output,
    accepts_config,
    accepts_run_manager,
)
from langchain_core.tracers.langchain import LangChainTracer

if TYPE_CHECKING:
    from langchain_core.callbacks.base import BaseCallbackManager, Callbacks
    from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForChainRun,
        CallbackManagerForChainRun,
    )
else:
    # Pydantic validates through typed dicts, but
    # the callbacks need forward refs updated
    Callbacks = list | Any | None


class EmptyDict(TypedDict, total=False):
    """Empty dict type."""


class RunnableConfig(TypedDict, total=False):
    """Configuration for a `Runnable`.

    !!! note Custom values

        The `TypedDict` has `total=False` set intentionally to:

        - Allow partial configs to be created and merged together via `merge_configs`
        - Support config propagation from parent to child runnables via
            `var_child_runnable_config` (a `ContextVar` that automatically passes
            config down the call stack without explicit parameter passing), where
            configs are merged rather than replaced

        !!! example

            ```python
            # Parent sets tags
            chain.invoke(input, config={"tags": ["parent"]})
            # Child automatically inherits and can add:
            # ensure_config({"tags": ["child"]}) -> {"tags": ["parent", "child"]}
            ```
    """

    tags: list[str]
    """Tags for this call and any sub-calls (e.g. a Chain calling an LLM).

    You can use these to filter calls.
    """

    metadata: dict[str, Any]
    """Metadata for this call and any sub-calls (e.g. a Chain calling an LLM).

    Keys should be strings, values should be JSON-serializable.
    """

    callbacks: Callbacks
    """Callbacks for this call and any sub-calls (e.g. a Chain calling an LLM).

    Tags are passed to all callbacks, metadata is passed to handle*Start callbacks.
    """

    run_name: str
    """Name for the tracer run for this call.

    Defaults to the name of the class."""

    max_concurrency: int | None
    """Maximum number of parallel calls to make.

    If not provided, defaults to `ThreadPoolExecutor`'s default.
    """

    recursion_limit: int
    """Maximum number of times a call can recurse.

    If not provided, defaults to `25`.
    """

    configurable: dict[str, Any]
    """Runtime values for attributes previously made configurable on this `Runnable`,
    or sub-`Runnable` objects, through `configurable_fields` or
    `configurable_alternatives`.

    Check `output_schema` for a description of the attributes that have been made
    configurable.
    """

    run_id: uuid.UUID | None
    """Unique identifier for the tracer run for this call.

    If not provided, a new UUID will be generated.
    """


CONFIG_KEYS = [
    "tags",
    "metadata",
    "callbacks",
    "run_name",
    "max_concurrency",
    "recursion_limit",
    "configurable",
    "run_id",
]

COPIABLE_KEYS = [
    "tags",
    "metadata",
    "callbacks",
    "configurable",
]

DEFAULT_RECURSION_LIMIT = 25


var_child_runnable_config: ContextVar[RunnableConfig | None] = ContextVar(
    "child_runnable_config", default=None
)


# This is imported and used in langgraph, so don't break.
def _set_config_context(
    config: RunnableConfig,
) -> tuple[Token[RunnableConfig | None], dict[str, Any] | None]:
    """Set the child Runnable config + tracing context.

    Args:
        config: The config to set.

    Returns:
        The token to reset the config and the previous tracing context.
    """
    config_token = var_child_runnable_config.set(config)
    current_context = None
    if (
        (callbacks := config.get("callbacks"))
        and (
            parent_run_id := getattr(callbacks, "parent_run_id", None)
        )  # Is callback manager
        and (
            tracer := next(
                (
                    handler
                    for handler in getattr(callbacks, "handlers", [])
                    if isinstance(handler, LangChainTracer)
                ),
                None,
            )
        )
        and (run := tracer.run_map.get(str(parent_run_id)))
    ):
        current_context = get_tracing_context()
        _set_tracing_context({"parent": run})
    return config_token, current_context


@contextmanager
def set_config_context(config: RunnableConfig) -> Generator[Context, None, None]:
    """Set the child Runnable config + tracing context.

    Args:
        config: The config to set.

    Yields:
        The config context.
    """
    ctx = copy_context()
    config_token, _ = ctx.run(_set_config_context, config)
    try:
        yield ctx
    finally:
        ctx.run(var_child_runnable_config.reset, config_token)
        ctx.run(
            _set_tracing_context,
            {
                "parent": None,
                "project_name": None,
                "tags": None,
                "metadata": None,
                "enabled": None,
                "client": None,
            },
        )


def ensure_config(config: RunnableConfig | None = None) -> RunnableConfig:
    """Ensure that a config is a dict with all keys present.

    Args:
        config: The config to ensure.

    Returns:
        The ensured config.
    """
    empty = RunnableConfig(
        tags=[],
        metadata={},
        callbacks=None,
        recursion_limit=DEFAULT_RECURSION_LIMIT,
        configurable={},
    )
    if var_config := var_child_runnable_config.get():
        empty.update(
            cast(
                "RunnableConfig",
                {
                    k: v.copy() if k in COPIABLE_KEYS else v  # type: ignore[attr-defined]
                    for k, v in var_config.items()
                    if v is not None
                },
            )
        )
    if config is not None:
        empty.update(
            cast(
                "RunnableConfig",
                {
                    k: v.copy() if k in COPIABLE_KEYS else v  # type: ignore[attr-defined]
                    for k, v in config.items()
                    if v is not None and k in CONFIG_KEYS
                },
            )
        )
    if config is not None:
        for k, v in config.items():
            if k not in CONFIG_KEYS and v is not None:
                empty["configurable"][k] = v
    for key, value in empty.get("configurable", {}).items():
        if (
            not key.startswith("__")
            and isinstance(value, (str, int, float, bool))
            and key not in empty["metadata"]
            and key != "api_key"
        ):
            empty["metadata"][key] = value
    return empty


def get_config_list(
    config: RunnableConfig | Sequence[RunnableConfig] | None, length: int
) -> list[RunnableConfig]:
    """Get a list of configs from a single config or a list of configs.

     It is useful for subclasses overriding batch() or abatch().

    Args:
        config: The config or list of configs.
        length: The length of the list.

    Returns:
        The list of configs.

    Raises:
        ValueError: If the length of the list is not equal to the length of the inputs.

    """
    if length < 0:
        msg = f"length must be >= 0, but got {length}"
        raise ValueError(msg)
    if isinstance(config, Sequence) and len(config) != length:
        msg = (
            f"config must be a list of the same length as inputs, "
            f"but got {len(config)} configs for {length} inputs"
        )
        raise ValueError(msg)

    if isinstance(config, Sequence):
        return list(map(ensure_config, config))
    if length > 1 and isinstance(config, dict) and config.get("run_id") is not None:
        warnings.warn(
            "Provided run_id be used only for the first element of the batch.",
            category=RuntimeWarning,
            stacklevel=3,
        )
        subsequent = cast(
            "RunnableConfig", {k: v for k, v in config.items() if k != "run_id"}
        )
        return [
            ensure_config(subsequent) if i else ensure_config(config)
            for i in range(length)
        ]
    return [ensure_config(config) for i in range(length)]


def patch_config(
    config: RunnableConfig | None,
    *,
    callbacks: BaseCallbackManager | None = None,
    recursion_limit: int | None = None,
    max_concurrency: int | None = None,
    run_name: str | None = None,
    configurable: dict[str, Any] | None = None,
) -> RunnableConfig:
    """Patch a config with new values.

    Args:
        config: The config to patch.
        callbacks: The callbacks to set.
        recursion_limit: The recursion limit to set.
        max_concurrency: The max concurrency to set.
        run_name: The run name to set.
        configurable: The configurable to set.

    Returns:
        The patched config.
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


def merge_configs(*configs: RunnableConfig | None) -> RunnableConfig:
    """Merge multiple configs into one.

    Args:
        *configs: The configs to merge.

    Returns:
        The merged config.
    """
    base: RunnableConfig = {}
    # Even though the keys aren't literals, this is correct
    # because both dicts are the same type
    for config in (ensure_config(c) for c in configs if c is not None):
        for key in config:
            if key == "metadata":
                base["metadata"] = {
                    **base.get("metadata", {}),
                    **(config.get("metadata") or {}),
                }
            elif key == "tags":
                base["tags"] = sorted(
                    set(base.get("tags", []) + (config.get("tags") or [])),
                )
            elif key == "configurable":
                base["configurable"] = {
                    **base.get("configurable", {}),
                    **(config.get("configurable") or {}),
                }
            elif key == "callbacks":
                base_callbacks = base.get("callbacks")
                these_callbacks = config["callbacks"]
                # callbacks can be either None, list[handler] or manager
                # so merging two callbacks values has 6 cases
                if isinstance(these_callbacks, list):
                    if base_callbacks is None:
                        base["callbacks"] = these_callbacks.copy()
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
                        base["callbacks"] = these_callbacks.copy()
                    elif isinstance(base_callbacks, list):
                        mngr = these_callbacks.copy()
                        for callback in base_callbacks:
                            mngr.add_handler(callback, inherit=True)
                        base["callbacks"] = mngr
                    else:
                        # base_callbacks is also a manager
                        base["callbacks"] = base_callbacks.merge(these_callbacks)
            elif key == "recursion_limit":
                if config["recursion_limit"] != DEFAULT_RECURSION_LIMIT:
                    base["recursion_limit"] = config["recursion_limit"]
            elif key in COPIABLE_KEYS and config[key] is not None:  # type: ignore[literal-required]
                base[key] = config[key].copy()  # type: ignore[literal-required]
            else:
                base[key] = config[key] or base.get(key)  # type: ignore[literal-required]
    return base


def call_func_with_variable_args(
    func: Callable[[Input], Output]
    | Callable[[Input, RunnableConfig], Output]
    | Callable[[Input, CallbackManagerForChainRun], Output]
    | Callable[[Input, CallbackManagerForChainRun, RunnableConfig], Output],
    input: Input,
    config: RunnableConfig,
    run_manager: CallbackManagerForChainRun | None = None,
    **kwargs: Any,
) -> Output:
    """Call function that may optionally accept a run_manager and/or config.

    Args:
        func: The function to call.
        input: The input to the function.
        config: The config to pass to the function.
        run_manager: The run manager to pass to the function.
        **kwargs: The keyword arguments to pass to the function.

    Returns:
        The output of the function.
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
    func: Callable[[Input], Awaitable[Output]]
    | Callable[[Input, RunnableConfig], Awaitable[Output]]
    | Callable[[Input, AsyncCallbackManagerForChainRun], Awaitable[Output]]
    | Callable[
        [Input, AsyncCallbackManagerForChainRun, RunnableConfig], Awaitable[Output]
    ],
    input: Input,
    config: RunnableConfig,
    run_manager: AsyncCallbackManagerForChainRun | None = None,
    **kwargs: Any,
) -> Awaitable[Output]:
    """Async call function that may optionally accept a run_manager and/or config.

    Args:
        func: The function to call.
        input: The input to the function.
        config: The config to pass to the function.
        run_manager: The run manager to pass to the function.
        **kwargs: The keyword arguments to pass to the function.

    Returns:
        The output of the function.
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
        config: The config.

    Returns:
        The callback manager.
    """
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
        config: The config.

    Returns:
        The async callback manager.
    """
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
            func: The function to submit.
            *args: The positional arguments to the function.
            **kwargs: The keyword arguments to the function.

        Returns:
            The future for the function.
        """
        return super().submit(
            cast("Callable[..., T]", partial(copy_context().run, func, *args, **kwargs))
        )

    def map(
        self,
        fn: Callable[..., T],
        *iterables: Iterable[Any],
        **kwargs: Any,
    ) -> Iterator[T]:
        """Map a function to multiple iterables.

        Args:
            fn: The function to map.
            *iterables: The iterables to map over.
            timeout: The timeout for the map.
            chunksize: The chunksize for the map.

        Returns:
            The iterator for the mapped function.
        """
        contexts = [copy_context() for _ in range(len(iterables[0]))]  # type: ignore[arg-type]

        def _wrapped_fn(*args: Any) -> T:
            return contexts.pop().run(fn, *args)

        return super().map(
            _wrapped_fn,
            *iterables,
            **kwargs,
        )


@contextmanager
def get_executor_for_config(
    config: RunnableConfig | None,
) -> Generator[Executor, None, None]:
    """Get an executor for a config.

    Args:
        config: The config.

    Yields:
        The executor.
    """
    config = config or {}
    with ContextThreadPoolExecutor(
        max_workers=config.get("max_concurrency")
    ) as executor:
        yield executor


async def run_in_executor(
    executor_or_config: Executor | RunnableConfig | None,
    func: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    """Run a function in an executor.

    Args:
        executor_or_config: The executor or config to run in.
        func: The function.
        *args: The positional arguments to the function.
        **kwargs: The keyword arguments to the function.

    Returns:
        The output of the function.
    """

    def wrapper() -> T:
        try:
            return func(*args, **kwargs)
        except StopIteration as exc:
            # StopIteration can't be set on an asyncio.Future
            # it raises a TypeError and leaves the Future pending forever
            # so we need to convert it to a RuntimeError
            raise RuntimeError from exc

    if executor_or_config is None or isinstance(executor_or_config, dict):
        # Use default executor with context copied from current context
        return await asyncio.get_running_loop().run_in_executor(
            None,
            cast("Callable[..., T]", partial(copy_context().run, wrapper)),
        )

    return await asyncio.get_running_loop().run_in_executor(executor_or_config, wrapper)
