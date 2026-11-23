"""Process-global default `TracePolicy` for agent middleware.

`TracePolicy` (from langgraph) shapes what a middleware hook span records. This module
adds a process-wide default that individual middleware override via their own
`trace_policy` attribute. Set it once at startup with `configure_trace_policy`; the
default is read at span-creation time, so it applies to agents built before or after
the call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from langgraph.types import TracePolicy

if TYPE_CHECKING:
    from collections.abc import Callable

_DEFAULT_TRACE_POLICY: TracePolicy | None = None


def configure_trace_policy(policy: TracePolicy | None) -> None:
    """Set the process-wide default `TracePolicy` for agent middleware hook spans.

    Call once at startup. A middleware's own `trace_policy` overrides this wholesale
    (no field-level merge). Pass `None` to clear the default.
    """
    global _DEFAULT_TRACE_POLICY  # noqa: PLW0603 - process-wide default, set once at startup
    _DEFAULT_TRACE_POLICY = policy


def _resolve(mw_policy: TracePolicy | None) -> TracePolicy | None:
    """The effective policy: the middleware's own if set, else the global default."""
    return mw_policy if mw_policy is not None else _DEFAULT_TRACE_POLICY


def _resolved_transform(
    mw_policy: TracePolicy | None,
    field: Literal["process_inputs", "process_outputs"],
) -> Callable[[Any], Any]:
    """A trace processor that resolves the effective policy at call time.

    Reading the global at call time (not build time) means `configure_trace_policy` can
    run after `create_agent` and still apply. Passes the value through unchanged when no
    effective policy defines `field`.
    """

    def process(value: Any) -> Any:
        effective = _resolve(mw_policy)
        fn = getattr(effective, field) if effective is not None else None
        return fn(value) if fn is not None else value

    return process


def _node_trace_policy(mw_policy: TracePolicy | None) -> TracePolicy:
    """Build the resolver-backed `TracePolicy` for a middleware's node hook."""
    return TracePolicy(
        process_inputs=_resolved_transform(mw_policy, "process_inputs"),
        process_outputs=_resolved_transform(mw_policy, "process_outputs"),
    )
