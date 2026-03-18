"""Tests for CiscoAIDefenseMiddleware and CiscoAIDefenseToolMiddleware."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime

from langchain.agents.middleware.cisco_ai_defense import (
    CiscoAIDefenseError,
    CiscoAIDefenseMiddleware,
    CiscoAIDefenseToolMiddleware,
)
from langchain.agents.middleware.types import AgentState


# ---------------------------------------------------------------------------
# Fakes for aidefense API responses
# ---------------------------------------------------------------------------

@dataclass
class FakeInspectResponse:
    is_safe: bool
    classifications: list[str] = field(default_factory=list)


@dataclass
class FakeMCPInspectResult:
    is_safe: bool
    classifications: list[str] = field(default_factory=list)


@dataclass
class FakeMCPInspectResponse:
    result: FakeMCPInspectResult | None = None
    error: Any = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state_with_messages(messages: list) -> AgentState:
    return {"messages": messages}


def _make_runtime() -> Runtime:
    return Runtime()


def _mock_chat_client(*, is_safe: bool, classifications: list[str] | None = None):
    client = MagicMock()
    client.inspect_conversation.return_value = FakeInspectResponse(
        is_safe=is_safe,
        classifications=classifications or [],
    )
    return client


def _mock_mcp_client(*, is_safe: bool):
    client = MagicMock()
    safe_result = FakeMCPInspectResponse(
        result=FakeMCPInspectResult(is_safe=is_safe),
    )
    client.inspect_tool_call.return_value = safe_result
    client.inspect_response.return_value = safe_result
    return client


# ===================================================================
# CiscoAIDefenseMiddleware tests
# ===================================================================


class TestCiscoAIDefenseMiddleware:
    """Unit tests for the LLM inspection middleware."""

    def _middleware(self, **overrides: Any) -> CiscoAIDefenseMiddleware:
        defaults: dict[str, Any] = {
            "api_key": "test-key",
            "region": "us",
            "exit_behavior": "end",
        }
        defaults.update(overrides)
        return CiscoAIDefenseMiddleware(**defaults)

    def test_before_model_safe_input_passes(self) -> None:
        mw = self._middleware()
        mw._client = _mock_chat_client(is_safe=True)

        state = _state_with_messages([HumanMessage("hello")])
        result = mw.before_model(state, _make_runtime())

        assert result is None

    def test_before_model_unsafe_input_blocks(self) -> None:
        mw = self._middleware()
        mw._client = _mock_chat_client(
            is_safe=False, classifications=["prompt_injection"],
        )

        state = _state_with_messages([HumanMessage("DROP TABLE users")])
        result = mw.before_model(state, _make_runtime())

        assert result is not None
        assert result["jump_to"] == "end"
        assert len(result["messages"]) == 1
        assert "blocked" in result["messages"][0].content.lower()
        assert "prompt_injection" in result["messages"][0].content

    def test_before_model_unsafe_input_raises(self) -> None:
        mw = self._middleware(exit_behavior="error")
        mw._client = _mock_chat_client(
            is_safe=False, classifications=["jailbreak"],
        )

        state = _state_with_messages([HumanMessage("bad")])
        with pytest.raises(CiscoAIDefenseError, match="blocked"):
            mw.before_model(state, _make_runtime())

    def test_after_model_safe_output_passes(self) -> None:
        mw = self._middleware()
        mw._client = _mock_chat_client(is_safe=True)

        state = _state_with_messages([
            HumanMessage("hello"),
            AIMessage("Hi there!"),
        ])
        result = mw.after_model(state, _make_runtime())

        assert result is None

    def test_after_model_unsafe_output_blocks(self) -> None:
        mw = self._middleware()
        mw._client = _mock_chat_client(
            is_safe=False, classifications=["pii_leak"],
        )

        state = _state_with_messages([
            HumanMessage("hello"),
            AIMessage("Here is the SSN: 123-45-6789"),
        ])
        result = mw.after_model(state, _make_runtime())

        assert result is not None
        assert result["jump_to"] == "end"
        assert "pii_leak" in result["messages"][0].content

    def test_off_mode_check_input_false(self) -> None:
        mw = self._middleware(check_input=False)
        mw._client = _mock_chat_client(is_safe=False)

        state = _state_with_messages([HumanMessage("anything")])
        result = mw.before_model(state, _make_runtime())

        assert result is None
        mw._client.inspect_conversation.assert_not_called()

    def test_fail_open_on_api_error(self) -> None:
        mw = self._middleware(fail_open=True)
        mw._client = MagicMock()
        mw._client.inspect_conversation.side_effect = ConnectionError("timeout")

        state = _state_with_messages([HumanMessage("hello")])
        result = mw.before_model(state, _make_runtime())

        assert result is None

    def test_fail_closed_on_api_error(self) -> None:
        mw = self._middleware(fail_open=False)
        mw._client = MagicMock()
        mw._client.inspect_conversation.side_effect = ConnectionError("timeout")

        state = _state_with_messages([HumanMessage("hello")])
        with pytest.raises(ConnectionError, match="timeout"):
            mw.before_model(state, _make_runtime())

    def test_invalid_exit_behavior(self) -> None:
        with pytest.raises(ValueError, match="Invalid exit_behavior"):
            self._middleware(exit_behavior="invalid")

    def test_region_normalization(self) -> None:
        mw = self._middleware(region="eu")
        assert mw.region == "eu-central-1"

        mw2 = self._middleware(region="apj")
        assert mw2.region == "ap-northeast-1"

        mw3 = self._middleware(region="us-east-1")
        assert mw3.region == "us-east-1"


# ===================================================================
# CiscoAIDefenseToolMiddleware tests
# ===================================================================


class TestCiscoAIDefenseToolMiddleware:
    """Unit tests for the tool/MCP inspection middleware."""

    def _middleware(self, **overrides: Any) -> CiscoAIDefenseToolMiddleware:
        defaults: dict[str, Any] = {
            "api_key": "test-key",
            "region": "us",
            "exit_behavior": "end",
        }
        defaults.update(overrides)
        return CiscoAIDefenseToolMiddleware(**defaults)

    @staticmethod
    def _make_request(
        name: str = "my_tool",
        args: dict | None = None,
        call_id: str = "call-1",
    ) -> MagicMock:
        req = MagicMock()
        req.tool_call = {
            "name": name,
            "args": args or {},
            "id": call_id,
        }
        return req

    @staticmethod
    def _make_handler(content: str = "tool result") -> MagicMock:
        handler = MagicMock()
        handler.return_value = ToolMessage(content=content, tool_call_id="call-1")
        return handler

    def test_tool_safe_request_passes(self) -> None:
        mw = self._middleware()
        mw._client = _mock_mcp_client(is_safe=True)

        request = self._make_request()
        handler = self._make_handler()

        result = mw.wrap_tool_call(request, handler)

        handler.assert_called_once_with(request)
        assert isinstance(result, ToolMessage)
        assert result.content == "tool result"

    def test_tool_unsafe_request_blocks(self) -> None:
        mw = self._middleware()
        mw._client = _mock_mcp_client(is_safe=False)

        request = self._make_request(name="exec_cmd")
        handler = self._make_handler()

        result = mw.wrap_tool_call(request, handler)

        handler.assert_not_called()
        assert isinstance(result, ToolMessage)
        assert "blocked" in result.content.lower()
        assert result.tool_call_id == "call-1"

    def test_tool_unsafe_response_blocks(self) -> None:
        mw = self._middleware(inspect_requests=False)
        safe_client = MagicMock()
        unsafe_resp = FakeMCPInspectResponse(
            result=FakeMCPInspectResult(is_safe=False),
        )
        safe_client.inspect_response.return_value = unsafe_resp
        mw._client = safe_client

        request = self._make_request()
        handler = self._make_handler(content="SSN: 123-45-6789")

        result = mw.wrap_tool_call(request, handler)

        handler.assert_called_once()
        assert isinstance(result, ToolMessage)
        assert "blocked" in result.content.lower()

    def test_tool_inspect_requests_only(self) -> None:
        mw = self._middleware(inspect_responses=False)
        mw._client = _mock_mcp_client(is_safe=True)

        request = self._make_request()
        handler = self._make_handler()

        result = mw.wrap_tool_call(request, handler)

        mw._client.inspect_tool_call.assert_called_once()
        mw._client.inspect_response.assert_not_called()
        assert result.content == "tool result"

    def test_tool_fail_open(self) -> None:
        mw = self._middleware(fail_open=True)
        mw._client = MagicMock()
        mw._client.inspect_tool_call.side_effect = ConnectionError("timeout")

        request = self._make_request()
        handler = self._make_handler()

        result = mw.wrap_tool_call(request, handler)

        handler.assert_called_once()
        assert result.content == "tool result"

    def test_tool_fail_closed(self) -> None:
        mw = self._middleware(fail_open=False)
        mw._client = MagicMock()
        mw._client.inspect_tool_call.side_effect = ConnectionError("timeout")

        request = self._make_request()
        handler = self._make_handler()

        with pytest.raises(ConnectionError, match="timeout"):
            mw.wrap_tool_call(request, handler)

    def test_tool_unsafe_request_raises_error(self) -> None:
        mw = self._middleware(exit_behavior="error")
        mw._client = _mock_mcp_client(is_safe=False)

        request = self._make_request(name="danger_tool")
        handler = self._make_handler()

        with pytest.raises(CiscoAIDefenseError, match="blocked"):
            mw.wrap_tool_call(request, handler)

        handler.assert_not_called()
