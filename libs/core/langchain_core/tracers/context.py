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

from langchain_core.tracers.core import _tracing_error_handlers
from langchain_core.tracers.langchain import LangChainTracer
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from langsmith import Client as LangSmithClient
    from langsmith.address import AgentAddress

    from langchain_core.callbacks.base import BaseCallbackHandler, Callbacks
    from langchain_core.callbacks.manager import AsyncCallbackManager, CallbackManager
    from langchain_core.tracers.schemas import Run

# for backwards partial compatibility if this is imported by users but unused
tracing_callback_var: Any = None
tracing_v2_callback_var: ContextVar[LangChainTracer | None] = ContextVar(
    "tracing_callback_v2", default=None
)
run_collector_var: ContextVar[RunCollectorCallbackHandler | None] = ContextVar(
    "run_collector", default=None
)


@contextmanager
def tracing_error_handler(
    handler: Callable[[BaseException, Run], None],
) -> Generator[None, None, None]:
    """Customize errored tracer runs before they are persisted.

    The synchronous handler receives the original exception and the live `Run`
    after its error, end time, events, and any partial outputs have been assigned.
    It may mutate the run, for example to add exception-specific metadata.
    This applies to existing and newly configured sync and async tracers that use
    the base tracer error lifecycle; it does not enable tracing itself.

    Nested handlers run from outermost to innermost. Ordinary handler exceptions
    are logged and suppressed so remaining handlers and persistence can proceed.
    The previous handlers are restored on exit, including exceptional exit.
    Handlers are local to the current context and inherited by child tasks and
    LangChain's context-propagating executors, not arbitrary threads. Child tasks
    retain their inherited context even if they outlive this context manager.

    Args:
        handler: Synchronous callback accepting the original exception and run.
            It must not change run identity or parentage and should not block.

    Yields:
        `None`.
    """
    token = _tracing_error_handlers.set((*_tracing_error_handlers.get(), handler))
    try:
        yield
    finally:
        _tracing_error_handlers.reset(token)


@contextmanager
def tracing_v2_enabled(
    project_name: str | None = None,
    *,
    example_id: str | UUID | None = None,
    tags: list[str] | None = None,
    client: LangSmithClient | None = None,
    address: AgentAddress | None = None,
) -> Generator[LangChainTracer, None, None]:
    """Instruct LangChain to log all runs in context to LangSmith.

    Args:
        project_name: The name of the project.

            Resolved together with `address` from LangSmith config and the
            environment when no destination is supplied.
        example_id: The ID of the example.
        tags: The tags to add to the run.
        client: The client of the langsmith.
        address: The agent address to send traces to instead of a project.

            !!! warning "Beta"
                Agent addressing must be enabled for the LangSmith workspace.

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
        address=address,
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
    *,
    address: AgentAddress | None = None,
) -> Callbacks:
    if _tracing_v2_is_enabled():
        tracer = tracing_v2_callback_var.get() or LangChainTracer(
            example_id=example_id,
            **_get_tracer_kwargs(project_name, address=address),
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


def _get_tracer_kwargs(
    project_name: str | None = None, *, address: AgentAddress | None = None
) -> dict[str, Any]:
    """Get tracer defaults from the current LangSmith context.

    Forward project and address together so an addressed parent never picks up
    an ambient project. Leave environment defaults and validation to RunTree.
    """
    context = ls_rh.get_tracing_context()
    parent = context["parent"]
    if project_name is not None or address is not None:
        project = project_name
    elif parent is not None:
        project, address = parent.session_name, parent.address
    else:
        project, address = context["project_name"], context["address"]
    return {
        "project_name": project,
        "address": address,
        "client": parent.client if parent is not None else context["client"],
        "tags": context["tags"],
        "metadata": context["metadata"],
    }


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
