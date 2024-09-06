"""Test CallbackManager."""

import functools
from unittest.mock import patch

import pytest
from langchain_core.callbacks.manager import CallbackManager, trace_as_chain_group
from langchain_core.outputs import LLMResult
from langchain_core.tracers.langchain import LangChainTracer, wait_for_all_tracers
from langchain_core.utils.pydantic import get_fields

from langchain_community.callbacks import get_openai_callback
from langchain_community.callbacks.manager import get_bedrock_anthropic_callback
from langchain_community.llms.openai import BaseOpenAI


@pytest.fixture(autouse=True)
def disable_lru_cache(monkeypatch):
    """Disable lru_cache for tests. (Env caching in langfuse)"""

    def no_cache_decorator(func=None, *, maxsize=None, typed=False):
        if func is not None:
            return func
        return lambda f: f

    monkeypatch.setattr(functools, "lru_cache", no_cache_decorator)


def test_callback_manager_configure_context_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test callback manager configuration."""
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGSMITH_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING", "false")
    monkeypatch.setenv("LANGSMITH_TRACING_V2", "false")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "foo")
    monkeypatch.setenv("LANGCHAIN_TRACING", "foo")
    with patch.object(LangChainTracer, "_update_run_single"):
        with patch.object(LangChainTracer, "_persist_run_single"):
            with trace_as_chain_group("test") as group_manager:
                assert len(group_manager.handlers) == 1
                tracer = group_manager.handlers[0]
                assert isinstance(tracer, LangChainTracer)

                with get_openai_callback() as cb:
                    # This is a new empty callback handler
                    assert cb.successful_requests == 0
                    assert cb.total_tokens == 0

                    # configure adds this openai cb but doesn't modify the group manager
                    mngr = CallbackManager.configure(group_manager)
                    assert mngr.handlers == [tracer, cb]
                    assert group_manager.handlers == [tracer]

                    response = LLMResult(
                        generations=[],
                        llm_output={
                            "token_usage": {
                                "prompt_tokens": 2,
                                "completion_tokens": 1,
                                "total_tokens": 3,
                            },
                            "model_name": get_fields(BaseOpenAI)["model_name"].default,
                        },
                    )
                    mngr.on_llm_start({}, ["prompt"])[0].on_llm_end(response)

                    # The callback handler has been updated
                    assert cb.successful_requests == 1
                    assert cb.total_tokens == 3
                    assert cb.prompt_tokens == 2
                    assert cb.completion_tokens == 1
                    assert cb.total_cost > 0

                with get_openai_callback() as cb:
                    # This is a new empty callback handler
                    assert cb.successful_requests == 0
                    assert cb.total_tokens == 0

                    # configure adds this openai cb but doesn't modify the group manager
                    mngr = CallbackManager.configure(group_manager)
                    assert mngr.handlers == [tracer, cb]
                    assert group_manager.handlers == [tracer]

                    response = LLMResult(
                        generations=[],
                        llm_output={
                            "token_usage": {
                                "prompt_tokens": 2,
                                "completion_tokens": 1,
                                "total_tokens": 3,
                            },
                            "model_name": get_fields(BaseOpenAI)["model_name"].default,
                        },
                    )
                    mngr.on_llm_start({}, ["prompt"])[0].on_llm_end(response)

                    # The callback handler has been updated
                    assert cb.successful_requests == 1
                    assert cb.total_tokens == 3
                    assert cb.prompt_tokens == 2
                    assert cb.completion_tokens == 1
                    assert cb.total_cost > 0

                with get_bedrock_anthropic_callback() as cb:
                    # This is a new empty callback handler
                    assert cb.successful_requests == 0
                    assert cb.total_tokens == 0

                    # configure adds this bedrock anthropic cb,
                    # but doesn't modify the group manager
                    mngr = CallbackManager.configure(group_manager)
                    assert mngr.handlers == [tracer, cb]
                    assert group_manager.handlers == [tracer]

                    response = LLMResult(
                        generations=[],
                        llm_output={
                            "usage": {
                                "prompt_tokens": 2,
                                "completion_tokens": 1,
                                "total_tokens": 3,
                            },
                            "model_id": "anthropic.claude-instant-v1",
                        },
                    )
                    mngr.on_llm_start({}, ["prompt"])[0].on_llm_end(response)

                    # The callback handler has been updated
                    assert cb.successful_requests == 1
                    assert cb.total_tokens == 3
                    assert cb.prompt_tokens == 2
                    assert cb.completion_tokens == 1
                    assert cb.total_cost > 0
            wait_for_all_tracers()
            assert LangChainTracer._persist_run_single.call_count == 1  # type: ignore
