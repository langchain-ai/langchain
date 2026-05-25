"""Unit tests for MCP trust verification and MCPToolkit."""

from __future__ import annotations

import os
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain_anthropic.mcp.toolkit import MCPToolkit, _CallableTrustVerifier
from langchain_anthropic.mcp.trust import (
    DominionObservatoryVerifier,
    TrustFailureMode,
    TrustScore,
    TrustVerificationError,
    TrustVerifier,
)

# Prevent any real Anthropic API calls during tests
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

# --------------------------------------------------------------------------- #
# Shared fixtures / helpers
# --------------------------------------------------------------------------- #

_SERVER_URL = "https://mcp.example.com/mcp"
_SERVER_URL_2 = "https://mcp2.example.com/mcp"
_PASSING_API_RESPONSE = {"trust_score": 0.9, "sla_grade": "A"}
_FAILING_API_RESPONSE = {"trust_score": 0.5, "sla_grade": "F"}

_MCP_SERVERS = [{"type": "url", "url": _SERVER_URL, "name": "example-mcp"}]


def _mock_sync_client(
    data: dict[str, Any],
    status_code: int = 200,
    raise_on_get: Exception | None = None,
) -> MagicMock:
    """Build a mock httpx.Client context manager."""
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = data
    if status_code >= 400:
        import httpx

        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=response
        )
    else:
        response.raise_for_status = MagicMock()

    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    if raise_on_get:
        client.get.side_effect = raise_on_get
    else:
        client.get.return_value = response
    return client


def _mock_async_client(
    data: dict[str, Any],
    status_code: int = 200,
    raise_on_get: Exception | None = None,
) -> AsyncMock:
    """Build a mock httpx.AsyncClient context manager."""
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = data
    if status_code >= 400:
        import httpx

        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=response
        )
    else:
        response.raise_for_status = MagicMock()

    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    if raise_on_get:
        client.get = AsyncMock(side_effect=raise_on_get)
    else:
        client.get = AsyncMock(return_value=response)
    return client


# --------------------------------------------------------------------------- #
# TrustVerifier abstract protocol
# --------------------------------------------------------------------------- #


class TestTrustVerifierAbstract:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            TrustVerifier()  # type: ignore[abstract]

    def test_concrete_subclass_works(self) -> None:
        class SimpleVerifier(TrustVerifier):
            def verify(self, server_url: str) -> TrustScore:
                return TrustScore(trust_score=1.0, sla_grade="A")

            async def averify(self, server_url: str) -> TrustScore:
                return self.verify(server_url)

        v = SimpleVerifier()
        score = v.verify(_SERVER_URL)
        assert score.trust_score == 1.0
        assert score.sla_grade == "A"


# --------------------------------------------------------------------------- #
# DominionObservatoryVerifier — synchronous
# --------------------------------------------------------------------------- #


class TestDominionVerifierSync:
    def _make(self, **kwargs: Any) -> DominionObservatoryVerifier:
        return DominionObservatoryVerifier(**kwargs)

    def test_passes_when_score_above_threshold(self) -> None:
        v = self._make(trust_threshold=0.7)
        client = _mock_sync_client(_PASSING_API_RESPONSE)
        with patch("httpx.Client", return_value=client):
            score = v.verify(_SERVER_URL)
        assert score.trust_score == 0.9
        assert score.sla_grade == "A"

    def test_raises_when_score_below_threshold(self) -> None:
        v = self._make(trust_threshold=0.7)
        client = _mock_sync_client(_FAILING_API_RESPONSE)
        with patch("httpx.Client", return_value=client):
            with pytest.raises(TrustVerificationError, match="below threshold"):
                v.verify(_SERVER_URL)

    def test_unreachable_allow_mode_passes(self) -> None:
        v = self._make(trust_failure_mode=TrustFailureMode.ALLOW)
        client = _mock_sync_client({}, raise_on_get=ConnectionError("unreachable"))
        with patch("httpx.Client", return_value=client):
            score = v.verify(_SERVER_URL)
        assert score.trust_score == 1.0
        assert score.sla_grade == "N/A"

    def test_unreachable_deny_mode_raises(self) -> None:
        v = self._make(trust_failure_mode=TrustFailureMode.DENY)
        client = _mock_sync_client({}, raise_on_get=ConnectionError("unreachable"))
        with patch("httpx.Client", return_value=client):
            with pytest.raises(TrustVerificationError, match="DENY failure mode"):
                v.verify(_SERVER_URL)

    def test_ttl_cache_prevents_duplicate_calls(self) -> None:
        v = self._make(trust_threshold=0.7, ttl=300)
        client = _mock_sync_client(_PASSING_API_RESPONSE)
        with patch("httpx.Client", return_value=client):
            v.verify(_SERVER_URL)
            v.verify(_SERVER_URL)
            v.verify(_SERVER_URL)
        assert client.get.call_count == 1

    def test_ttl_cache_expires_and_refreshes(self) -> None:
        v = self._make(trust_threshold=0.7, ttl=60)
        # Seed with an already-expired entry
        v._cache[_SERVER_URL] = (
            TrustScore(trust_score=0.9, sla_grade="A"),
            time.monotonic() - 1,
        )
        client = _mock_sync_client(_PASSING_API_RESPONSE)
        with patch("httpx.Client", return_value=client):
            v.verify(_SERVER_URL)
        assert client.get.call_count == 1

    def test_warm_cache_skips_api(self) -> None:
        v = self._make(trust_threshold=0.7, ttl=300)
        v._cache[_SERVER_URL] = (
            TrustScore(trust_score=0.95, sla_grade="A+"),
            time.monotonic() + 300,
        )
        client = _mock_sync_client(_PASSING_API_RESPONSE)
        with patch("httpx.Client", return_value=client):
            score = v.verify(_SERVER_URL)
        client.get.assert_not_called()
        assert score.trust_score == 0.95


# --------------------------------------------------------------------------- #
# DominionObservatoryVerifier — asynchronous
# --------------------------------------------------------------------------- #


class TestDominionVerifierAsync:
    def _make(self, **kwargs: Any) -> DominionObservatoryVerifier:
        return DominionObservatoryVerifier(**kwargs)

    @pytest.mark.asyncio
    async def test_passes_when_score_above_threshold(self) -> None:
        v = self._make(trust_threshold=0.7)
        client = _mock_async_client(_PASSING_API_RESPONSE)
        with patch("httpx.AsyncClient", return_value=client):
            score = await v.averify(_SERVER_URL)
        assert score.trust_score == 0.9
        assert score.sla_grade == "A"

    @pytest.mark.asyncio
    async def test_raises_when_score_below_threshold(self) -> None:
        v = self._make(trust_threshold=0.7)
        client = _mock_async_client(_FAILING_API_RESPONSE)
        with patch("httpx.AsyncClient", return_value=client):
            with pytest.raises(TrustVerificationError, match="below threshold"):
                await v.averify(_SERVER_URL)

    @pytest.mark.asyncio
    async def test_unreachable_allow_mode_passes(self) -> None:
        v = self._make(trust_failure_mode=TrustFailureMode.ALLOW)
        client = _mock_async_client({}, raise_on_get=ConnectionError("unreachable"))
        with patch("httpx.AsyncClient", return_value=client):
            score = await v.averify(_SERVER_URL)
        assert score.trust_score == 1.0
        assert score.sla_grade == "N/A"

    @pytest.mark.asyncio
    async def test_unreachable_deny_mode_raises(self) -> None:
        v = self._make(trust_failure_mode=TrustFailureMode.DENY)
        client = _mock_async_client({}, raise_on_get=ConnectionError("unreachable"))
        with patch("httpx.AsyncClient", return_value=client):
            with pytest.raises(TrustVerificationError, match="DENY failure mode"):
                await v.averify(_SERVER_URL)

    @pytest.mark.asyncio
    async def test_ttl_cache_prevents_duplicate_calls(self) -> None:
        v = self._make(trust_threshold=0.7, ttl=300)
        client = _mock_async_client(_PASSING_API_RESPONSE)
        with patch("httpx.AsyncClient", return_value=client):
            await v.averify(_SERVER_URL)
            await v.averify(_SERVER_URL)
            await v.averify(_SERVER_URL)
        assert client.get.call_count == 1

    @pytest.mark.asyncio
    async def test_ttl_cache_expires_and_refreshes(self) -> None:
        v = self._make(trust_threshold=0.7, ttl=60)
        v._cache[_SERVER_URL] = (
            TrustScore(trust_score=0.9, sla_grade="A"),
            time.monotonic() - 1,
        )
        client = _mock_async_client(_PASSING_API_RESPONSE)
        with patch("httpx.AsyncClient", return_value=client):
            await v.averify(_SERVER_URL)
        assert client.get.call_count == 1

    @pytest.mark.asyncio
    async def test_warm_cache_skips_api(self) -> None:
        v = self._make(trust_threshold=0.7, ttl=300)
        v._cache[_SERVER_URL] = (
            TrustScore(trust_score=0.95, sla_grade="A+"),
            time.monotonic() + 300,
        )
        client = _mock_async_client(_PASSING_API_RESPONSE)
        with patch("httpx.AsyncClient", return_value=client):
            score = await v.averify(_SERVER_URL)
        client.get.assert_not_called()
        assert score.trust_score == 0.95


# --------------------------------------------------------------------------- #
# MCPToolkit — synchronous
# --------------------------------------------------------------------------- #


class TestMCPToolkitSync:
    def _passing_verifier(self) -> TrustVerifier:
        v = MagicMock(spec=TrustVerifier)
        v.verify.return_value = TrustScore(trust_score=0.9, sla_grade="A")
        return v

    def _failing_verifier(self) -> TrustVerifier:
        v = MagicMock(spec=TrustVerifier)
        v.verify.side_effect = TrustVerificationError("score too low")
        return v

    def test_score_above_threshold_passes_tool_execution(self) -> None:
        toolkit = MCPToolkit(
            servers=_MCP_SERVERS,
            trust_verifier=self._passing_verifier(),
        )
        with patch("langchain_anthropic.chat_models.ChatAnthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            model = toolkit.get_model(model="claude-sonnet-4-6")
        mock_cls.assert_called_once()
        assert model is mock_cls.return_value

    def test_score_below_threshold_blocks_tool_execution(self) -> None:
        toolkit = MCPToolkit(
            servers=_MCP_SERVERS,
            trust_verifier=self._failing_verifier(),
        )
        with pytest.raises(TrustVerificationError, match="score too low"):
            toolkit.get_model(model="claude-sonnet-4-6")

    def test_unreachable_allow_mode_allows_execution(self) -> None:
        v = MagicMock(spec=TrustVerifier)
        v.verify.return_value = TrustScore(trust_score=1.0, sla_grade="N/A")
        toolkit = MCPToolkit(
            servers=_MCP_SERVERS,
            trust_failure_mode=TrustFailureMode.ALLOW,
            trust_verifier=v,
        )
        with patch("langchain_anthropic.chat_models.ChatAnthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            model = toolkit.get_model(model="claude-sonnet-4-6")
        assert model is mock_cls.return_value

    def test_unreachable_deny_mode_blocks_execution(self) -> None:
        v = MagicMock(spec=TrustVerifier)
        v.verify.side_effect = TrustVerificationError("API unreachable, DENY")
        toolkit = MCPToolkit(
            servers=_MCP_SERVERS,
            trust_failure_mode=TrustFailureMode.DENY,
            trust_verifier=v,
        )
        with pytest.raises(TrustVerificationError):
            toolkit.get_model(model="claude-sonnet-4-6")

    def test_custom_callable_verifier_is_used(self) -> None:
        calls: list[str] = []

        def my_verifier(server_url: str) -> TrustScore:
            calls.append(server_url)
            return TrustScore(trust_score=0.95, sla_grade="A")

        toolkit = MCPToolkit(servers=_MCP_SERVERS, trust_verifier=my_verifier)
        with patch("langchain_anthropic.chat_models.ChatAnthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            toolkit.get_model(model="claude-sonnet-4-6")

        assert calls == [_SERVER_URL]
        # Ensure callable is wrapped in _CallableTrustVerifier
        assert isinstance(toolkit._verifier, _CallableTrustVerifier)

    def test_default_verifier_used_when_none_provided(self) -> None:
        toolkit = MCPToolkit(servers=_MCP_SERVERS)
        assert isinstance(toolkit._verifier, DominionObservatoryVerifier)

    def test_custom_trust_verifier_instance_is_used_directly(self) -> None:
        custom = self._passing_verifier()
        toolkit = MCPToolkit(servers=_MCP_SERVERS, trust_verifier=custom)
        assert toolkit._verifier is custom

    def test_verifier_called_per_server_url(self) -> None:
        multi_servers = [
            {"type": "url", "url": _SERVER_URL, "name": "s1"},
            {"type": "url", "url": _SERVER_URL_2, "name": "s2"},
        ]
        v = self._passing_verifier()
        toolkit = MCPToolkit(servers=multi_servers, trust_verifier=v)
        with patch("langchain_anthropic.chat_models.ChatAnthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            toolkit.get_model(model="claude-sonnet-4-6")
        assert v.verify.call_count == 2
        v.verify.assert_any_call(_SERVER_URL)
        v.verify.assert_any_call(_SERVER_URL_2)

    def test_ttl_cache_prevents_duplicate_api_calls(self) -> None:
        toolkit = MCPToolkit(servers=_MCP_SERVERS, trust_threshold=0.7, ttl=300)
        client = _mock_sync_client(_PASSING_API_RESPONSE)
        with patch("httpx.Client", return_value=client):
            with patch("langchain_anthropic.chat_models.ChatAnthropic") as mock_cls:
                mock_cls.return_value = MagicMock()
                toolkit.get_model(model="claude-sonnet-4-6")
                toolkit.get_model(model="claude-sonnet-4-6")
        assert client.get.call_count == 1

    def test_ttl_cache_expires_and_makes_fresh_call(self) -> None:
        toolkit = MCPToolkit(servers=_MCP_SERVERS, trust_threshold=0.7, ttl=60)
        assert isinstance(toolkit._verifier, DominionObservatoryVerifier)
        toolkit._verifier._cache[_SERVER_URL] = (
            TrustScore(trust_score=0.9, sla_grade="A"),
            time.monotonic() - 1,
        )
        client = _mock_sync_client(_PASSING_API_RESPONSE)
        with patch("httpx.Client", return_value=client):
            with patch("langchain_anthropic.chat_models.ChatAnthropic") as mock_cls:
                mock_cls.return_value = MagicMock()
                toolkit.get_model(model="claude-sonnet-4-6")
        assert client.get.call_count == 1


# --------------------------------------------------------------------------- #
# MCPToolkit — asynchronous
# --------------------------------------------------------------------------- #


class TestMCPToolkitAsync:
    def _passing_verifier(self) -> AsyncMock:
        v = AsyncMock(spec=TrustVerifier)
        v.verify.return_value = TrustScore(trust_score=0.9, sla_grade="A")
        v.averify = AsyncMock(return_value=TrustScore(trust_score=0.9, sla_grade="A"))
        return v

    def _failing_verifier(self) -> AsyncMock:
        v = AsyncMock(spec=TrustVerifier)
        v.averify = AsyncMock(side_effect=TrustVerificationError("score too low"))
        return v

    @pytest.mark.asyncio
    async def test_score_above_threshold_passes_tool_execution(self) -> None:
        toolkit = MCPToolkit(
            servers=_MCP_SERVERS,
            trust_verifier=self._passing_verifier(),
        )
        with patch("langchain_anthropic.chat_models.ChatAnthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            model = await toolkit.aget_model(model="claude-sonnet-4-6")
        mock_cls.assert_called_once()
        assert model is mock_cls.return_value

    @pytest.mark.asyncio
    async def test_score_below_threshold_blocks_tool_execution(self) -> None:
        toolkit = MCPToolkit(
            servers=_MCP_SERVERS,
            trust_verifier=self._failing_verifier(),
        )
        with pytest.raises(TrustVerificationError, match="score too low"):
            await toolkit.aget_model(model="claude-sonnet-4-6")

    @pytest.mark.asyncio
    async def test_unreachable_allow_mode_allows_execution(self) -> None:
        v = self._passing_verifier()
        v.averify = AsyncMock(return_value=TrustScore(trust_score=1.0, sla_grade="N/A"))
        toolkit = MCPToolkit(
            servers=_MCP_SERVERS,
            trust_failure_mode=TrustFailureMode.ALLOW,
            trust_verifier=v,
        )
        with patch("langchain_anthropic.chat_models.ChatAnthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            model = await toolkit.aget_model(model="claude-sonnet-4-6")
        assert model is mock_cls.return_value

    @pytest.mark.asyncio
    async def test_unreachable_deny_mode_blocks_execution(self) -> None:
        v = AsyncMock(spec=TrustVerifier)
        v.averify = AsyncMock(side_effect=TrustVerificationError("DENY"))
        toolkit = MCPToolkit(
            servers=_MCP_SERVERS,
            trust_failure_mode=TrustFailureMode.DENY,
            trust_verifier=v,
        )
        with pytest.raises(TrustVerificationError):
            await toolkit.aget_model(model="claude-sonnet-4-6")

    @pytest.mark.asyncio
    async def test_custom_callable_verifier_is_used(self) -> None:
        calls: list[str] = []

        def my_verifier(server_url: str) -> TrustScore:
            calls.append(server_url)
            return TrustScore(trust_score=0.95, sla_grade="A")

        toolkit = MCPToolkit(servers=_MCP_SERVERS, trust_verifier=my_verifier)
        with patch("langchain_anthropic.chat_models.ChatAnthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            await toolkit.aget_model(model="claude-sonnet-4-6")

        assert calls == [_SERVER_URL]

    @pytest.mark.asyncio
    async def test_default_verifier_used_when_none_provided(self) -> None:
        toolkit = MCPToolkit(servers=_MCP_SERVERS)
        assert isinstance(toolkit._verifier, DominionObservatoryVerifier)

    @pytest.mark.asyncio
    async def test_ttl_cache_prevents_duplicate_api_calls(self) -> None:
        toolkit = MCPToolkit(servers=_MCP_SERVERS, trust_threshold=0.7, ttl=300)
        client = _mock_async_client(_PASSING_API_RESPONSE)
        with patch("httpx.AsyncClient", return_value=client):
            with patch("langchain_anthropic.chat_models.ChatAnthropic") as mock_cls:
                mock_cls.return_value = MagicMock()
                await toolkit.aget_model(model="claude-sonnet-4-6")
                await toolkit.aget_model(model="claude-sonnet-4-6")
        assert client.get.call_count == 1

    @pytest.mark.asyncio
    async def test_ttl_cache_expires_and_makes_fresh_call(self) -> None:
        toolkit = MCPToolkit(servers=_MCP_SERVERS, trust_threshold=0.7, ttl=60)
        assert isinstance(toolkit._verifier, DominionObservatoryVerifier)
        toolkit._verifier._cache[_SERVER_URL] = (
            TrustScore(trust_score=0.9, sla_grade="A"),
            time.monotonic() - 1,
        )
        client = _mock_async_client(_PASSING_API_RESPONSE)
        with patch("httpx.AsyncClient", return_value=client):
            with patch("langchain_anthropic.chat_models.ChatAnthropic") as mock_cls:
                mock_cls.return_value = MagicMock()
                await toolkit.aget_model(model="claude-sonnet-4-6")
        assert client.get.call_count == 1
