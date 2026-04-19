"""Private constants shared across tracer and runnable-config code.

These live in their own module so they can be imported without pulling in
the heavier ``langchain_core.tracers.langchain`` dependency (which loads
``langsmith`` transitively).
"""

from __future__ import annotations

LANGSMITH_INHERITABLE_METADATA_KEYS: frozenset[str] = frozenset(("ls_agent_type",))
"""Allowlist of metadata keys routed to LangSmith tracers only.

Keys in this set are:

1. Stripped from general ``inheritable_metadata`` by
   ``langchain_core.runnables.config._split_inheritable_metadata`` so they
   don't reach non-tracer callback handlers (``stream_events``,
   ``astream_log``, user-provided ``BaseCallbackHandler`` instances, etc.).
2. Forwarded to ``LangChainTracer`` as *overridable* defaults via
   ``LangChainTracer.copy_with_metadata_defaults``. Unlike general
   metadata defaults (first-wins), keys in this allowlist are last-wins so
   that a nested ``RunnableConfig`` / ``CallbackManager.configure`` call
   can rescope the value to the innermost run (e.g. ``ls_agent_type``).
"""
# TODO: Expand this to cover all `ls_`-prefixed metadata keys.
