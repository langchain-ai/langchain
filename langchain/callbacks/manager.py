from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
)
from uuid import UUID

import langchain
from langchain.callbacks.openai_info import OpenAICallbackHandler
from langchain.callbacks.stdout import StdOutCallbackHandler
from langchain.callbacks.tracers.langchain import LangChainTracer
from langchain.callbacks.tracers.langchain_v1 import LangChainTracerV1, TracerSessionV1
from langchain.callbacks.tracers.stdout import ConsoleCallbackHandler
from langchain.callbacks.tracers.wandb import WandbTracer
from langchain.schema.callbacks.manager import (
    AsyncCallbackManager,
    # For backwards compatability
    AsyncCallbackManagerForChainRun,  # noqa: F401
    CallbackManager,
    # For backwards compatability
    CallbackManagerForChainRun,  # noqa: F401
    Callbacks,
)

logger = logging.getLogger(__name__)

openai_callback_var: ContextVar[Optional[OpenAICallbackHandler]] = ContextVar(
    "openai_callback", default=None
)
tracing_callback_var: ContextVar[
    Optional[LangChainTracerV1]
] = ContextVar(  # noqa: E501
    "tracing_callback", default=None
)
wandb_tracing_callback_var: ContextVar[
    Optional[WandbTracer]
] = ContextVar(  # noqa: E501
    "tracing_wandb_callback", default=None
)

tracing_v2_callback_var: ContextVar[
    Optional[LangChainTracer]
] = ContextVar(  # noqa: E501
    "tracing_callback_v2", default=None
)


def _get_debug() -> bool:
    return langchain.debug


@contextmanager
def get_openai_callback() -> Generator[OpenAICallbackHandler, None, None]:
    """Get the OpenAI callback handler in a context manager.
    which conveniently exposes token and cost information.

    Returns:
        OpenAICallbackHandler: The OpenAI callback handler.

    Example:
        >>> with get_openai_callback() as cb:
        ...     # Use the OpenAI callback handler
    """
    cb = OpenAICallbackHandler()
    openai_callback_var.set(cb)
    yield cb
    openai_callback_var.set(None)


@contextmanager
def tracing_enabled(
    session_name: str = "default",
) -> Generator[TracerSessionV1, None, None]:
    """Get the Deprecated LangChainTracer in a context manager.

    Args:
        session_name (str, optional): The name of the session.
          Defaults to "default".

    Returns:
        TracerSessionV1: The LangChainTracer session.

    Example:
        >>> with tracing_enabled() as session:
        ...     # Use the LangChainTracer session
    """
    cb = LangChainTracerV1()
    session = cast(TracerSessionV1, cb.load_session(session_name))
    tracing_callback_var.set(cb)
    yield session
    tracing_callback_var.set(None)


@contextmanager
def wandb_tracing_enabled(
    session_name: str = "default",
) -> Generator[None, None, None]:
    """Get the WandbTracer in a context manager.

    Args:
        session_name (str, optional): The name of the session.
            Defaults to "default".

    Returns:
        None

    Example:
        >>> with wandb_tracing_enabled() as session:
        ...     # Use the WandbTracer session
    """
    cb = WandbTracer()
    wandb_tracing_callback_var.set(cb)
    yield None
    wandb_tracing_callback_var.set(None)


@contextmanager
def tracing_v2_enabled(
    project_name: Optional[str] = None,
    *,
    example_id: Optional[Union[str, UUID]] = None,
    tags: Optional[List[str]] = None,
) -> Generator[None, None, None]:
    """Instruct LangChain to log all runs in context to LangSmith.

    Args:
        project_name (str, optional): The name of the project.
            Defaults to "default".
        example_id (str or UUID, optional): The ID of the example.
            Defaults to None.
        tags (List[str], optional): The tags to add to the run.
            Defaults to None.

    Returns:
        None

    Example:
        >>> with tracing_v2_enabled():
        ...     # LangChain code will automatically be traced
    """
    if isinstance(example_id, str):
        example_id = UUID(example_id)
    cb = LangChainTracer(
        example_id=example_id,
        project_name=project_name,
        tags=tags,
    )
    tracing_v2_callback_var.set(cb)
    yield
    tracing_v2_callback_var.set(None)


@contextmanager
def trace_as_chain_group(
    group_name: str,
    *,
    project_name: Optional[str] = None,
    example_id: Optional[Union[str, UUID]] = None,
    tags: Optional[List[str]] = None,
) -> Generator[CallbackManager, None, None]:
    """Get a callback manager for a chain group in a context manager.
    Useful for grouping different calls together as a single run even if
    they aren't composed in a single chain.

    Args:
        group_name (str): The name of the chain group.
        project_name (str, optional): The name of the project.
            Defaults to None.
        example_id (str or UUID, optional): The ID of the example.
            Defaults to None.
        tags (List[str], optional): The inheritable tags to apply to all runs.
            Defaults to None.

    Returns:
        CallbackManager: The callback manager for the chain group.

    Example:
        >>> with trace_as_chain_group("group_name") as manager:
        ...     # Use the callback manager for the chain group
        ...     llm.predict("Foo", callbacks=manager)
    """
    cb = LangChainTracer(
        project_name=project_name,
        example_id=example_id,
    )
    cm = CallbackManager.configure(
        inheritable_callbacks=[cb],
        inheritable_tags=tags,
    )

    run_manager = cm.on_chain_start({"name": group_name}, {})
    yield run_manager.get_child()
    run_manager.on_chain_end({})


@asynccontextmanager
async def atrace_as_chain_group(
    group_name: str,
    *,
    project_name: Optional[str] = None,
    example_id: Optional[Union[str, UUID]] = None,
    tags: Optional[List[str]] = None,
) -> AsyncGenerator[AsyncCallbackManager, None]:
    """Get an async callback manager for a chain group in a context manager.
    Useful for grouping different async calls together as a single run even if
    they aren't composed in a single chain.

    Args:
        group_name (str): The name of the chain group.
        project_name (str, optional): The name of the project.
            Defaults to None.
        example_id (str or UUID, optional): The ID of the example.
            Defaults to None.
        tags (List[str], optional): The inheritable tags to apply to all runs.
            Defaults to None.
    Returns:
        AsyncCallbackManager: The async callback manager for the chain group.

    Example:
        >>> async with atrace_as_chain_group("group_name") as manager:
        ...     # Use the async callback manager for the chain group
        ...     await llm.apredict("Foo", callbacks=manager)
    """
    cb = LangChainTracer(
        project_name=project_name,
        example_id=example_id,
    )
    cm = AsyncCallbackManager.configure(
        inheritable_callbacks=[cb], inheritable_tags=tags
    )

    run_manager = await cm.on_chain_start({"name": group_name}, {})
    try:
        yield run_manager.get_child()
    finally:
        await run_manager.on_chain_end({})


T = TypeVar("T", CallbackManager, AsyncCallbackManager)


def env_var_is_set(env_var: str) -> bool:
    """Check if an environment variable is set.

    Args:
        env_var (str): The name of the environment variable.

    Returns:
        bool: True if the environment variable is set, False otherwise.
    """
    return env_var in os.environ and os.environ[env_var] not in (
        "",
        "0",
        "false",
        "False",
    )


def _configure(
    callback_manager_cls: Type[T],
    inheritable_callbacks: Callbacks = None,
    local_callbacks: Callbacks = None,
    verbose: bool = False,
    inheritable_tags: Optional[List[str]] = None,
    local_tags: Optional[List[str]] = None,
    inheritable_metadata: Optional[Dict[str, Any]] = None,
    local_metadata: Optional[Dict[str, Any]] = None,
) -> T:
    """Configure the callback manager.

    Args:
        callback_manager_cls (Type[T]): The callback manager class.
        inheritable_callbacks (Optional[Callbacks], optional): The inheritable
            callbacks. Defaults to None.
        local_callbacks (Optional[Callbacks], optional): The local callbacks.
            Defaults to None.
        verbose (bool, optional): Whether to enable verbose mode. Defaults to False.
        inheritable_tags (Optional[List[str]], optional): The inheritable tags.
            Defaults to None.
        local_tags (Optional[List[str]], optional): The local tags. Defaults to None.
        inheritable_metadata (Optional[Dict[str, Any]], optional): The inheritable
            metadata. Defaults to None.
        local_metadata (Optional[Dict[str, Any]], optional): The local metadata.
            Defaults to None.

    Returns:
        T: The configured callback manager.
    """
    callback_manager = callback_manager_cls(handlers=[])
    if inheritable_callbacks or local_callbacks:
        if isinstance(inheritable_callbacks, list) or inheritable_callbacks is None:
            inheritable_callbacks_ = inheritable_callbacks or []
            callback_manager = callback_manager_cls(
                handlers=inheritable_callbacks_.copy(),
                inheritable_handlers=inheritable_callbacks_.copy(),
            )
        else:
            callback_manager = callback_manager_cls(
                handlers=inheritable_callbacks.handlers,
                inheritable_handlers=inheritable_callbacks.inheritable_handlers,
                parent_run_id=inheritable_callbacks.parent_run_id,
                tags=inheritable_callbacks.tags,
                inheritable_tags=inheritable_callbacks.inheritable_tags,
                metadata=inheritable_callbacks.metadata,
                inheritable_metadata=inheritable_callbacks.inheritable_metadata,
            )
        local_handlers_ = (
            local_callbacks
            if isinstance(local_callbacks, list)
            else (local_callbacks.handlers if local_callbacks else [])
        )
        for handler in local_handlers_:
            callback_manager.add_handler(handler, False)
    if inheritable_tags or local_tags:
        callback_manager.add_tags(inheritable_tags or [])
        callback_manager.add_tags(local_tags or [], False)
    if inheritable_metadata or local_metadata:
        callback_manager.add_metadata(inheritable_metadata or {})
        callback_manager.add_metadata(local_metadata or {}, False)

    tracer = tracing_callback_var.get()
    wandb_tracer = wandb_tracing_callback_var.get()
    open_ai = openai_callback_var.get()
    tracing_enabled_ = (
        env_var_is_set("LANGCHAIN_TRACING")
        or tracer is not None
        or env_var_is_set("LANGCHAIN_HANDLER")
    )
    wandb_tracing_enabled_ = (
        env_var_is_set("LANGCHAIN_WANDB_TRACING") or wandb_tracer is not None
    )

    tracer_v2 = tracing_v2_callback_var.get()
    tracing_v2_enabled_ = (
        env_var_is_set("LANGCHAIN_TRACING_V2") or tracer_v2 is not None
    )
    tracer_project = os.environ.get(
        "LANGCHAIN_PROJECT", os.environ.get("LANGCHAIN_SESSION", "default")
    )
    debug = _get_debug()
    if (
        verbose
        or debug
        or tracing_enabled_
        or tracing_v2_enabled_
        or wandb_tracing_enabled_
        or open_ai is not None
    ):
        if verbose and not any(
            isinstance(handler, StdOutCallbackHandler)
            for handler in callback_manager.handlers
        ):
            if debug:
                pass
            else:
                callback_manager.add_handler(StdOutCallbackHandler(), False)
        if debug and not any(
            isinstance(handler, ConsoleCallbackHandler)
            for handler in callback_manager.handlers
        ):
            callback_manager.add_handler(ConsoleCallbackHandler(), True)
        if tracing_enabled_ and not any(
            isinstance(handler, LangChainTracerV1)
            for handler in callback_manager.handlers
        ):
            if tracer:
                callback_manager.add_handler(tracer, True)
            else:
                handler = LangChainTracerV1()
                handler.load_session(tracer_project)
                callback_manager.add_handler(handler, True)
        if wandb_tracing_enabled_ and not any(
            isinstance(handler, WandbTracer) for handler in callback_manager.handlers
        ):
            if wandb_tracer:
                callback_manager.add_handler(wandb_tracer, True)
            else:
                handler = WandbTracer()
                callback_manager.add_handler(handler, True)
        if tracing_v2_enabled_ and not any(
            isinstance(handler, LangChainTracer)
            for handler in callback_manager.handlers
        ):
            if tracer_v2:
                callback_manager.add_handler(tracer_v2, True)
            else:
                try:
                    handler = LangChainTracer(project_name=tracer_project)
                    callback_manager.add_handler(handler, True)
                except Exception as e:
                    logger.warning(
                        "Unable to load requested LangChainTracer."
                        " To disable this warning,"
                        " unset the  LANGCHAIN_TRACING_V2 environment variables.",
                        e,
                    )
        if open_ai is not None and not any(
            isinstance(handler, OpenAICallbackHandler)
            for handler in callback_manager.handlers
        ):
            callback_manager.add_handler(open_ai, True)
    return callback_manager
