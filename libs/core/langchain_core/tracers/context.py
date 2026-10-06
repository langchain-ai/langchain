"""Context management for tracers."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
)
from uuid import UUID

from langsmith import run_helpers as ls_rh
from langsmith import utils as ls_utils

from langchain_core.tracers.langchain import (
    LangChainTracer,
    _get_default_project_name,
)
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler

if TYPE_CHECKING:
    from collections.abc import Generator

    from langsmith import Client as LangSmithClient

    from langchain_core.callbacks.base import BaseCallbackHandler, Callbacks
    from langchain_core.callbacks.manager import AsyncCallbackManager, CallbackManager

# for backwards partial compatibility if this is imported by users but unused
tracing_callback_var: Any = None
tracing_v2_callback_var: ContextVar[LangChainTracer | None] = ContextVar(
    "tracing_callback_v2", default=None
)
run_collector_var: ContextVar[RunCollectorCallbackHandler | None] = ContextVar(
    "run_collector", default=None
)


@contextmanager
def tracing_v2_enabled(
    project_name: str | None = None,
    *,
    example_id: str | UUID | None = None,
    tags: list[str] | None = None,
    client: LangSmithClient | None = None,
) -> Generator[LangChainTracer, None, None]:
    """Instruct LangChain to log all runs in context to LangSmith.

    Args:
        project_name: The name of the project.

            Defaults to `'default'`.
        example_id: The ID of the example.
        tags: The tags to add to the run.
        client: The client of the langsmith.

    Yields:
        The LangChain tracer.

    Example:
        >>> with tracing_v2_enabled():
        ...     # LangChain code will automatically be traced

        You can use this to fetch the LangSmith run URL:

        >>> with tracing_v2_enabled() as cb:
        ...     chain.invoke("foo")
        ...     run_url = cb.get_run_url()
    """
    if isinstance(example_id, str):
        example_id = UUID(example_id)
    cb = LangChainTracer(
        example_id=example_id,
        project_name=project_name,
        tags=tags,
        client=client,
    )
    token = tracing_v2_callback_var.set(cb)
    try:
        yield cb
    finally:
        tracing_v2_callback_var.reset(token)


@contextmanager
def collect_runs() -> Generator[RunCollectorCallbackHandler, None, None]:
    """Collect all run traces in context.

    Yields:
        The run collector callback handler.

    Example:
        >>> with collect_runs() as runs_cb:
                chain.invoke("foo")
                run_id = runs_cb.traced_runs[0].id
    """
    cb = RunCollectorCallbackHandler()
    token = run_collector_var.set(cb)
    try:
        yield cb
    finally:
        run_collector_var.reset(token)


def _get_trace_callbacks(
    project_name: str | None = None,
    example_id: str | UUID | None = None,
    callback_manager: CallbackManager | AsyncCallbackManager | None = None,
) -> Callbacks:
    if _tracing_v2_is_enabled():
        address = None
        if not project_name:
            project_name, address = _get_tracer_destination()
        tracer = tracing_v2_callback_var.get() or LangChainTracer(
            project_name=project_name,
            example_id=example_id,
            address=address,
        )
        if callback_manager is None:
            cb = cast("Callbacks", [tracer])
        else:
            if not any(
                isinstance(handler, LangChainTracer)
                for handler in callback_manager.handlers
            ):
                callback_manager.add_handler(tracer)
                # If it already has a LangChainTracer, we don't need to add another one.
                # this would likely mess up the trace hierarchy.
            cb = callback_manager
    else:
        cb = None
    return cb


def _tracing_v2_is_enabled() -> bool | Literal["local"]:
    if tracing_v2_callback_var.get() is not None:
        return True
    return ls_utils.tracing_is_enabled()


def _get_tracer_project() -> str | None:
    return _get_tracer_destination()[0]


def _get_tracer_destination() -> tuple[str | None, str | None]:
    """Get the `(project, address)` langsmith's tracing context sends runs to.

    At most one is set. Both are `None` when the `LANGSMITH_AGENT_*` env vars
    address the runs and nothing is named in code.
    """
    tracing_context = ls_rh.get_tracing_context()
    run_tree = tracing_context["parent"]
    if run_tree is None:
        if tracing_context["project_name"] is not None:
            return cast("str", tracing_context["project_name"]), None
        # Only set on langsmith versions with agent addressing.
        if (address := tracing_context.get("address")) is not None:
            return None, address
        return _get_default_project_name(), None
    # Note, if people are trying to nest @traceable functions and the
    # tracing_v2_enabled context manager, this will likely mess up the
    # tree structure.
    return run_tree.session_name, getattr(run_tree, "address", None)


_configure_hooks: list[
    tuple[
        ContextVar[BaseCallbackHandler | None],
        bool,
        type[BaseCallbackHandler] | None,
        str | None,
    ]
] = []


def register_configure_hook(
    context_var: ContextVar[Any | None],
    inheritable: bool,  # noqa: FBT001
    handle_class: type[BaseCallbackHandler] | None = None,
    env_var: str | None = None,
) -> None:
    """Register a configure hook.

    Args:
        context_var: The context variable.
        inheritable: Whether the context variable is inheritable.
        handle_class: The callback handler class.
        env_var: The environment variable.

    Raises:
        ValueError: If `env_var` is set, `handle_class` must also be set to a non-`None`
            value.
    """
    if env_var is not None and handle_class is None:
        msg = "If env_var is set, handle_class must also be set to a non-None value."
        raise ValueError(msg)

    _configure_hooks.append(
        (
            # the typings of ContextVar do not have the generic arg set as covariant
            # so we have to cast it
            cast("ContextVar[BaseCallbackHandler | None]", context_var),
            inheritable,
            handle_class,
            env_var,
        )
    )


register_configure_hook(run_collector_var, inheritable=False)
