"""Unit tests for NomaGuardrailMiddleware."""

import uuid
from typing import TYPE_CHECKING, Any, cast

import pytest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from langchain.agents.middleware.noma_guardrails import NomaGuardrailMiddleware
from langchain.agents.middleware.types import AgentState

if TYPE_CHECKING:
    from langgraph.runtime import Runtime


def _fake_runtime() -> "Runtime":
    return cast("Runtime", object())


def _make_state(messages: list[AnyMessage]) -> AgentState[Any]:
    return AgentState(messages=messages)


def _fake_secret() -> str:
    return uuid.uuid4().hex


def test_before_agent_allows_when_action_allow() -> None:
    """Should allow when aggregatedAction is allow."""
    middleware = NomaGuardrailMiddleware(client_id="test", client_secret=_fake_secret())

    def fake_scan(_: dict[str, Any]) -> dict[str, Any]:
        return {
            "aggregatedAction": "allow",
            "aggregatedScanResult": True,
            "scanResult": [
                {
                    "role": "user",
                    "results": {"blockingDetector": {"result": True}},
                }
            ],
        }

    middleware._client.scan = fake_scan
    state = _make_state([HumanMessage(content="hello")])

    result = middleware.before_agent(state, _fake_runtime())

    assert result is None


def test_before_agent_anonymizes_on_mask_action() -> None:
    """Should anonymize when aggregatedAction is mask."""
    middleware = NomaGuardrailMiddleware(client_id="test", client_secret=_fake_secret())

    def fake_scan(_: dict[str, Any]) -> dict[str, Any]:
        return {
            "aggregatedAction": "mask",
            "scanResult": [
                {
                    "role": "user",
                    "results": {
                        "sensitiveData": {"ssn": {"result": True}},
                        "anonymizedContent": {"anonymized": "[REDACTED]"},
                    },
                }
            ],
        }

    middleware._client.scan = fake_scan
    state = _make_state([HumanMessage(content="my ssn is 123")])

    result = middleware.before_agent(state, _fake_runtime())

    assert result is not None
    assert result.get("jump_to") is None
    messages = result.get("messages")
    assert messages is not None
    assert messages[-1].content == "[REDACTED]"


def test_before_agent_blocks_on_block_action() -> None:
    """Should block when aggregatedAction is block."""
    middleware = NomaGuardrailMiddleware(client_id="test", client_secret=_fake_secret())

    def fake_scan(_: dict[str, Any]) -> dict[str, Any]:
        return {
            "aggregatedAction": "block",
            "scanResult": [
                {
                    "role": "user",
                    "results": {"blockingDetector": {"result": True}},
                }
            ],
        }

    middleware._client.scan = fake_scan
    state = _make_state([HumanMessage(content="bad prompt")])

    result = middleware.before_agent(state, _fake_runtime())

    assert result is not None
    assert result.get("jump_to") == "end"
    messages = result.get("messages")
    assert messages is not None
    assert isinstance(messages[-1], AIMessage)
    assert messages[-1].content.startswith("Request blocked by Noma guardrail")


def test_after_agent_skips_if_already_blocked() -> None:
    """Should not rescan when last message is already a block response."""
    middleware = NomaGuardrailMiddleware(client_id="test", client_secret=_fake_secret())
    called = False

    def fake_scan(_: dict[str, Any]) -> dict[str, Any]:
        nonlocal called
        called = True
        return {"aggregatedScanResult": False, "scanResult": []}

    middleware._client.scan = fake_scan
    state = _make_state(
        [
            HumanMessage(content="hello"),
            AIMessage(content="Request blocked by Noma guardrail: unsafe content detected"),
        ]
    )

    result = middleware.after_agent(state, _fake_runtime())

    assert result is None
    assert called is False


@pytest.mark.asyncio
async def test_abefore_agent_anonymizes_on_mask_action() -> None:
    """Async path should anonymize when aggregatedAction is mask."""
    middleware = NomaGuardrailMiddleware(client_id="test", client_secret=_fake_secret())

    async def fake_ascan(_: dict[str, Any]) -> dict[str, Any]:
        return {
            "aggregatedAction": "mask",
            "scanResult": [
                {
                    "role": "user",
                    "results": {
                        "sensitiveData": {"ssn": {"result": True}},
                        "anonymizedContent": {"anonymized": "[REDACTED]"},
                    },
                }
            ],
        }

    middleware._client.ascan = fake_ascan
    state = _make_state([HumanMessage(content="my ssn is 123")])

    result = await middleware.abefore_agent(state, _fake_runtime())

    assert result is not None
    messages = result.get("messages")
    assert messages is not None
    assert messages[-1].content == "[REDACTED]"
