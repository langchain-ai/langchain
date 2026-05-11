"""Regression tests for GitHub issue #37248.

configurable field contaminated with internal LangGraph __pregel_* /
checkpoint_* keys in ToolRuntime.config when accessed via AgentMiddleware
in langchain 1.2.16.

Expected: ToolRuntime.config["configurable"] contains only user-supplied keys.
Actual (regression): it also contains __pregel_call, __pregel_read, etc.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command

from langchain.agents.factory import create_agent
from langchain.agents.middleware.types import AgentMiddleware, ToolCallRequest
from tests.unit_tests.agents.model import FakeToolCallingModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_INTERNAL_KEY_PREFIXES = ("__pregel",)
_INTERNAL_KEY_EXACT = frozenset({"checkpoint_id", "checkpoint_map", "checkpoint_ns"})


def _internal_keys(configurable: dict) -> set[str]:
    return {
        k
        for k in configurable
        if any(k.startswith(p) for p in _INTERNAL_KEY_PREFIXES)
        or k in _INTERNAL_KEY_EXACT
    }


# ---------------------------------------------------------------------------
# Middleware under test
# ---------------------------------------------------------------------------


class CapturingMiddleware(AgentMiddleware):
    """Records the configurable dict the interceptor sees."""

    captured: dict = {}
    call_count: int = 0

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        self.captured.update(request.runtime.config.get("configurable", {}))
        self.call_count += 1
        return await handler(request)


# ---------------------------------------------------------------------------
# Tool under test
# ---------------------------------------------------------------------------


@tool
def get_db_config(db_config_id: str) -> str:
    """Get database config by ID."""
    return f"db_config_{db_config_id}"


# ---------------------------------------------------------------------------
# Regression tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_configurable_not_contaminated_in_middleware() -> None:
    """Issue #37248: user-supplied configurable keys must be the ONLY keys visible
    inside AgentMiddleware.awrap_tool_call — no internal __pregel_* / checkpoint_*
    keys should leak through.
    """
    middleware = CapturingMiddleware()
    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [{"args": {"db_config_id": "my_db"}, "id": "call_1", "name": "get_db_config"}],
                [],
            ]
        ),
        tools=[get_db_config],
        middleware=[middleware],
    )

    await agent.ainvoke(
        {"messages": [HumanMessage(content="hello")]},
        config={
            "configurable": {
                "db_config_id": "my_db",
                "user_id": "user_123",
            }
        },
    )

    assert middleware.call_count == 1, "Interceptor should have been called once"

    leaked = _internal_keys(middleware.captured)
    assert not leaked, (
        f"Internal LangGraph keys leaked into ToolRuntime.config['configurable']: "
        f"{sorted(leaked)}\n"
        f"Full configurable seen: {sorted(middleware.captured.keys())}\n"
        f"Expected only: ['db_config_id', 'user_id']"
    )

    assert "db_config_id" in middleware.captured, (
        "User-supplied 'db_config_id' missing from configurable"
    )
    assert "user_id" in middleware.captured, (
        "User-supplied 'user_id' missing from configurable"
    )
    assert middleware.captured["db_config_id"] == "my_db"
    assert middleware.captured["user_id"] == "user_123"


@pytest.mark.asyncio
async def test_configurable_user_keys_preserved_in_middleware() -> None:
    """Complement: user-supplied configurable values must be accessible and correct."""
    seen: dict[str, Any] = {}

    class AssertingMiddleware(AgentMiddleware):
        async def awrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
        ) -> ToolMessage | Command:
            configurable = request.runtime.config.get("configurable", {})
            seen.update(configurable)
            return await handler(request)

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [{"args": {"db_config_id": "prod_db"}, "id": "call_2", "name": "get_db_config"}],
                [],
            ]
        ),
        tools=[get_db_config],
        middleware=[AssertingMiddleware()],
    )

    await agent.ainvoke(
        {"messages": [HumanMessage(content="go")]},
        config={
            "configurable": {
                "db_config_id": "prod_db",
                "tenant": "acme",
            }
        },
    )

    assert seen.get("db_config_id") == "prod_db"
    assert seen.get("tenant") == "acme"


# ---------------------------------------------------------------------------
# T3: Direct unit test of _sanitize_tool_call_request_config
# SRS §2.3 FilterInternalConfigKeys + plan §4 D1-D3
# docs/lld/modules/factory-middleware-sanitize.md (Exposes table, sanitizer row)
# ---------------------------------------------------------------------------


def _make_request(configurable: dict) -> Any:
    """Build a minimal ToolCallRequest with the given configurable dict."""
    from langgraph.prebuilt.tool_node import ToolCallRequest, ToolRuntime

    rt = ToolRuntime(
        state={},
        context=None,
        config={"configurable": configurable},
        stream_writer=None,
        tools=[],
        tool_call_id="call_test",
        store=None,
    )
    return ToolCallRequest(
        tool_call={"name": "noop", "args": {}, "id": "call_test"},
        tool=None,
        state={},
        runtime=rt,
    )


def test_sanitize_strips_internal_keys() -> None:
    """T3: _sanitize_tool_call_request_config removes __pregel_* and checkpoint_* keys.

    Given: a ToolCallRequest whose runtime.config["configurable"] contains one user key,
    one __pregel_* key, and one exact-match checkpoint_* key.
    When: _sanitize_tool_call_request_config is called.
    Then: the returned request's configurable contains only the user key; the original
    request is not mutated; the returned object is a different instance.
    """
    from langchain.agents.factory import _sanitize_tool_call_request_config

    _SENTINEL = object()
    request = _make_request(
        {
            "user_key": "value",
            "__pregel_runtime": _SENTINEL,
            "checkpoint_id": _SENTINEL,
        }
    )
    original_configurable = dict(request.runtime.config["configurable"])

    result = _sanitize_tool_call_request_config(request)

    # Returns a different object (copy was made)
    assert result is not request

    # Only user key survives
    assert result.runtime.config["configurable"] == {"user_key": "value"}

    # Original is not mutated
    assert request.runtime.config["configurable"] == original_configurable


# ---------------------------------------------------------------------------
# T4: Fast-path — returns original object when configurable is already clean
# plan §4 D2 + docs/lld/modules/factory-middleware-sanitize.md (Invariants §2)
# ---------------------------------------------------------------------------


def test_sanitize_fast_path_returns_same_object() -> None:
    """T4: _sanitize_tool_call_request_config returns the same object when no internal
    keys are present (fast path — no allocation).

    Given: a ToolCallRequest whose runtime.config["configurable"] contains only user keys.
    When: _sanitize_tool_call_request_config is called.
    Then: the returned object is identical (is) to the input — no copy is made.
    """
    from langchain.agents.factory import _sanitize_tool_call_request_config

    request = _make_request({"db_config_id": "x", "tenant": "y"})

    result = _sanitize_tool_call_request_config(request)

    assert result is request
