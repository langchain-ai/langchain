from unittest.mock import patch

from langchain_core.callbacks.manager import CallbackManager, _configure
from langchain_core.tracers.context import _get_trace_callbacks
from langchain_core.tracers.langchain import LangChainTracer


def test_local_tracing_does_not_attach_a_posting_tracer() -> None:
    with patch(
        "langchain_core.tracers.context.ls_utils.tracing_is_enabled",
        return_value="local",
    ):
        assert _get_trace_callbacks() is None
        manager = _configure(CallbackManager)

    assert not any(isinstance(handler, LangChainTracer) for handler in manager.handlers)


def test_disabled_tracing_does_not_attach_a_posting_tracer() -> None:
    with patch(
        "langchain_core.tracers.context.ls_utils.tracing_is_enabled",
        return_value=False,
    ):
        assert _get_trace_callbacks() is None
        manager = _configure(CallbackManager)

    assert not any(isinstance(handler, LangChainTracer) for handler in manager.handlers)
