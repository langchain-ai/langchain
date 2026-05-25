"""Unit tests for MCP server trust verification."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain_core.tools.mcp.trust import (
    DominionObservatoryVerifier,
    TrustFailureMode,
    TrustScore,
    TrustVerificationError,
    TrustVerifier,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_SERVER_URL = "https://mcp.example.com"
_PASSING_RESPONSE = {"trust_score": 0.9, "sla_grade": "A"}
_FAILING_RESPONSE = {"trust_score": 0.5, "sla_grade": "F"}


def _make_httpx_response(data: dict[str, Any], status_code: int = 200) -> MagicMock:
    """Build a mock httpx.Response."""
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = data
    mock.raise_for_status = MagicMock()
    if status_code >= 400:
        import httpx

        mock.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=mock
        )
    return mock


def _make_async_httpx_response(data: dict[str, Any], status_code: int = 200) -> MagicMock:
    """Build a mock async httpx.Response."""
    mock = _make_httpx_response(data, status_code)
    return mock


# --------------------------------------------------------------------------- #
# Abstract protocol
# --------------------------------------------------------------------------- #


class TestTrustVerifierIsAbstract:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            TrustVerifier()  # type: ignore[abstract]

    def test_concrete_subclass_works(self) -> None:
        class ConcreteVerifier(TrustVerifier):
            def verify(self, server_url: str) -> TrustScore:
                return TrustScore(trust_score=1.0, sla_grade="A")

            async def averify(self, server_url: str) -> TrustScore:
                return self.verify(server_url)

        v = ConcreteVerifier()
        score = v.verify(_SERVER_URL)
        assert score.trust_score == 1.0
        assert score.sla_grade == "A"


# --------------------------------------------------------------------------- #
# TrustScore dataclass
# --------------------------------------------------------------------------- #


class TestTrustScore:
    def test_fields(self) -> None:
        score = TrustScore(trust_score=0.85, sla_grade="B")
        assert score.trust_score == 0.85
        assert score.sla_grade == "B"

    def test_equality(self) -> None:
        assert TrustScore(0.9, "A") == TrustScore(0.9, "A")
        assert TrustScore(0.9, "A") != TrustScore(0.8, "A")


# --------------------------------------------------------------------------- #
# DominionObservatoryVerifier — synchronous
# --------------------------------------------------------------------------- #


class TestDominionObservatoryVerifierSync:
    def _verifier(self, **kwargs: Any) -> DominionObservatoryVerifier:
        return DominionObservatoryVerifier(**kwargs)

    def test_passes_when_score_above_threshold(self) -> None:
        verifier = self._verifier(trust_threshold=0.7)
        response = _make_httpx_response(_PASSING_RESPONSE)
        client_mock = MagicMock()
        client_mock.__enter__ = MagicMock(return_value=client_mock)
        client_mock.__exit__ = MagicMock(return_value=False)
        client_mock.get.return_value = response

        with patch("httpx.Client", return_value=client_mock):
            score = verifier.verify(_SERVER_URL)

        assert score.trust_score == 0.9
        assert score.sla_grade == "A"

    def test_raises_when_score_below_threshold(self) -> None:
        verifier = self._verifier(trust_threshold=0.7)
        response = _make_httpx_response(_FAILING_RESPONSE)
        client_mock = MagicMock()
        client_mock.__enter__ = MagicMock(return_value=client_mock)
        client_mock.__exit__ = MagicMock(return_value=False)
        client_mock.get.return_value = response

        with patch("httpx.Client", return_value=client_mock):
            with pytest.raises(TrustVerificationError, match="below threshold"):
                verifier.verify(_SERVER_URL)

    def test_unreachable_allow_mode_passes(self) -> None:
        verifier = self._verifier(trust_failure_mode=TrustFailureMode.ALLOW)
        client_mock = MagicMock()
        client_mock.__enter__ = MagicMock(return_value=client_mock)
        client_mock.__exit__ = MagicMock(return_value=False)
        client_mock.get.side_effect = ConnectionError("unreachable")

        with patch("httpx.Client", return_value=client_mock):
            score = verifier.verify(_SERVER_URL)

        assert score.trust_score == 1.0
        assert score.sla_grade == "N/A"

    def test_unreachable_deny_mode_raises(self) -> None:
        verifier = self._verifier(trust_failure_mode=TrustFailureMode.DENY)
        client_mock = MagicMock()
        client_mock.__enter__ = MagicMock(return_value=client_mock)
        client_mock.__exit__ = MagicMock(return_value=False)
        client_mock.get.side_effect = ConnectionError("unreachable")

        with patch("httpx.Client", return_value=client_mock):
            with pytest.raises(TrustVerificationError, match="DENY failure mode"):
                verifier.verify(_SERVER_URL)

    def test_ttl_cache_prevents_duplicate_api_calls(self) -> None:
        verifier = self._verifier(trust_threshold=0.7, ttl=300)
        response = _make_httpx_response(_PASSING_RESPONSE)
        client_mock = MagicMock()
        client_mock.__enter__ = MagicMock(return_value=client_mock)
        client_mock.__exit__ = MagicMock(return_value=False)
        client_mock.get.return_value = response

        with patch("httpx.Client", return_value=client_mock):
            verifier.verify(_SERVER_URL)
            verifier.verify(_SERVER_URL)
            verifier.verify(_SERVER_URL)

        # API should only have been called once despite three verify() calls
        assert client_mock.get.call_count == 1

    def test_ttl_cache_expires_and_makes_fresh_call(self) -> None:
        verifier = self._verifier(trust_threshold=0.7, ttl=60)
        response = _make_httpx_response(_PASSING_RESPONSE)
        client_mock = MagicMock()
        client_mock.__enter__ = MagicMock(return_value=client_mock)
        client_mock.__exit__ = MagicMock(return_value=False)
        client_mock.get.return_value = response

        # Prime the cache with an already-expired entry
        verifier._cache[_SERVER_URL] = (
            TrustScore(trust_score=0.9, sla_grade="A"),
            time.monotonic() - 1,  # expired one second ago
        )

        with patch("httpx.Client", return_value=client_mock):
            verifier.verify(_SERVER_URL)

        # Expired cache should trigger a fresh API call
        assert client_mock.get.call_count == 1

    def test_cache_hit_does_not_call_api(self) -> None:
        verifier = self._verifier(trust_threshold=0.7, ttl=300)

        # Seed cache with a valid, non-expired entry
        verifier._cache[_SERVER_URL] = (
            TrustScore(trust_score=0.95, sla_grade="A+"),
            time.monotonic() + 300,
        )

        client_mock = MagicMock()
        with patch("httpx.Client", return_value=client_mock):
            score = verifier.verify(_SERVER_URL)

        client_mock.get.assert_not_called()
        assert score.trust_score == 0.95


# --------------------------------------------------------------------------- #
# DominionObservatoryVerifier — asynchronous
# --------------------------------------------------------------------------- #


class TestDominionObservatoryVerifierAsync:
    def _verifier(self, **kwargs: Any) -> DominionObservatoryVerifier:
        return DominionObservatoryVerifier(**kwargs)

    @pytest.mark.asyncio
    async def test_passes_when_score_above_threshold(self) -> None:
        verifier = self._verifier(trust_threshold=0.7)
        response = _make_async_httpx_response(_PASSING_RESPONSE)
        async_client_mock = AsyncMock()
        async_client_mock.__aenter__ = AsyncMock(return_value=async_client_mock)
        async_client_mock.__aexit__ = AsyncMock(return_value=False)
        async_client_mock.get = AsyncMock(return_value=response)

        with patch("httpx.AsyncClient", return_value=async_client_mock):
            score = await verifier.averify(_SERVER_URL)

        assert score.trust_score == 0.9
        assert score.sla_grade == "A"

    @pytest.mark.asyncio
    async def test_raises_when_score_below_threshold(self) -> None:
        verifier = self._verifier(trust_threshold=0.7)
        response = _make_async_httpx_response(_FAILING_RESPONSE)
        async_client_mock = AsyncMock()
        async_client_mock.__aenter__ = AsyncMock(return_value=async_client_mock)
        async_client_mock.__aexit__ = AsyncMock(return_value=False)
        async_client_mock.get = AsyncMock(return_value=response)

        with patch("httpx.AsyncClient", return_value=async_client_mock):
            with pytest.raises(TrustVerificationError, match="below threshold"):
                await verifier.averify(_SERVER_URL)

    @pytest.mark.asyncio
    async def test_unreachable_allow_mode_passes(self) -> None:
        verifier = self._verifier(trust_failure_mode=TrustFailureMode.ALLOW)
        async_client_mock = AsyncMock()
        async_client_mock.__aenter__ = AsyncMock(return_value=async_client_mock)
        async_client_mock.__aexit__ = AsyncMock(return_value=False)
        async_client_mock.get = AsyncMock(side_effect=ConnectionError("unreachable"))

        with patch("httpx.AsyncClient", return_value=async_client_mock):
            score = await verifier.averify(_SERVER_URL)

        assert score.trust_score == 1.0
        assert score.sla_grade == "N/A"

    @pytest.mark.asyncio
    async def test_unreachable_deny_mode_raises(self) -> None:
        verifier = self._verifier(trust_failure_mode=TrustFailureMode.DENY)
        async_client_mock = AsyncMock()
        async_client_mock.__aenter__ = AsyncMock(return_value=async_client_mock)
        async_client_mock.__aexit__ = AsyncMock(return_value=False)
        async_client_mock.get = AsyncMock(side_effect=ConnectionError("unreachable"))

        with patch("httpx.AsyncClient", return_value=async_client_mock):
            with pytest.raises(TrustVerificationError, match="DENY failure mode"):
                await verifier.averify(_SERVER_URL)

    @pytest.mark.asyncio
    async def test_ttl_cache_prevents_duplicate_api_calls(self) -> None:
        verifier = self._verifier(trust_threshold=0.7, ttl=300)
        response = _make_async_httpx_response(_PASSING_RESPONSE)
        async_client_mock = AsyncMock()
        async_client_mock.__aenter__ = AsyncMock(return_value=async_client_mock)
        async_client_mock.__aexit__ = AsyncMock(return_value=False)
        async_client_mock.get = AsyncMock(return_value=response)

        with patch("httpx.AsyncClient", return_value=async_client_mock):
            await verifier.averify(_SERVER_URL)
            await verifier.averify(_SERVER_URL)
            await verifier.averify(_SERVER_URL)

        assert async_client_mock.get.call_count == 1

    @pytest.mark.asyncio
    async def test_ttl_cache_expires_and_makes_fresh_call(self) -> None:
        verifier = self._verifier(trust_threshold=0.7, ttl=60)
        response = _make_async_httpx_response(_PASSING_RESPONSE)
        async_client_mock = AsyncMock()
        async_client_mock.__aenter__ = AsyncMock(return_value=async_client_mock)
        async_client_mock.__aexit__ = AsyncMock(return_value=False)
        async_client_mock.get = AsyncMock(return_value=response)

        # Seed cache with an expired entry
        verifier._cache[_SERVER_URL] = (
            TrustScore(trust_score=0.9, sla_grade="A"),
            time.monotonic() - 1,
        )

        with patch("httpx.AsyncClient", return_value=async_client_mock):
            await verifier.averify(_SERVER_URL)

        assert async_client_mock.get.call_count == 1

    @pytest.mark.asyncio
    async def test_cache_hit_does_not_call_api(self) -> None:
        verifier = self._verifier(trust_threshold=0.7, ttl=300)

        verifier._cache[_SERVER_URL] = (
            TrustScore(trust_score=0.95, sla_grade="A+"),
            time.monotonic() + 300,
        )

        async_client_mock = AsyncMock()
        with patch("httpx.AsyncClient", return_value=async_client_mock):
            score = await verifier.averify(_SERVER_URL)

        async_client_mock.get.assert_not_called()
        assert score.trust_score == 0.95
