"""Abstract TrustVerifier protocol and default implementation for MCP server trust."""

from __future__ import annotations

import abc
import enum
import time
from dataclasses import dataclass
from typing import Any

import httpx


class TrustVerificationError(Exception):
    """Raised when an MCP server fails trust verification.

    This can occur when the server's trust score is below the configured
    threshold, or when the trust API is unreachable and the failure mode
    is set to DENY.
    """


class TrustFailureMode(enum.Enum):
    """Behavior when the trust verification API is unreachable.

    Attributes:
        ALLOW: Permit tool execution even when the API cannot be contacted.
        DENY: Block tool execution when the API cannot be contacted.
    """

    ALLOW = "allow"
    DENY = "deny"


@dataclass
class TrustScore:
    """Trust assessment returned by a TrustVerifier for an MCP server.

    Attributes:
        trust_score: Numeric trust level in the range [0.0, 1.0], where
            1.0 represents full trust and 0.0 represents no trust.
        sla_grade: Qualitative grade label from the trust provider
            (e.g. ``"A"``, ``"B"``, ``"F"``).
    """

    trust_score: float
    sla_grade: str


class TrustVerifier(abc.ABC):
    """Abstract base class for MCP server trust verifiers.

    Implementations check whether an MCP server is trustworthy before
    tool execution proceeds. Both synchronous and asynchronous entry
    points are required so callers can choose based on their runtime.

    Example:
        .. code-block:: python

            class MyVerifier(TrustVerifier):
                def verify(self, server_url: str) -> TrustScore:
                    # custom logic
                    return TrustScore(trust_score=0.9, sla_grade="A")

                async def averify(self, server_url: str) -> TrustScore:
                    return self.verify(server_url)
    """

    @abc.abstractmethod
    def verify(self, server_url: str) -> TrustScore:
        """Verify the trustworthiness of an MCP server synchronously.

        Args:
            server_url: The URL of the MCP server to verify.

        Returns:
            `TrustScore` describing the server's trust level.

        Raises:
            TrustVerificationError: If the server does not meet the
                verifier's trust requirements.
        """

    @abc.abstractmethod
    async def averify(self, server_url: str) -> TrustScore:
        """Verify the trustworthiness of an MCP server asynchronously.

        Args:
            server_url: The URL of the MCP server to verify.

        Returns:
            `TrustScore` describing the server's trust level.

        Raises:
            TrustVerificationError: If the server does not meet the
                verifier's trust requirements.
        """


class DominionObservatoryVerifier(TrustVerifier):
    """Trust verifier backed by the Dominion Observatory public API.

    Calls ``GET https://dominionobservatory.com/api/trust?url=<server_url>``
    and returns a `TrustScore` from the response JSON. Results are cached
    per server URL for `ttl` seconds to stay within the 50 queries/day
    free-tier limit.

    Example:
        .. code-block:: python

            verifier = DominionObservatoryVerifier(
                trust_threshold=0.8,
                trust_failure_mode=TrustFailureMode.DENY,
                ttl=300,
            )
            score = verifier.verify("https://my-mcp-server.example.com")
    """

    _API_BASE = "https://dominionobservatory.com/api/trust"

    def __init__(
        self,
        *,
        trust_threshold: float = 0.7,
        trust_failure_mode: TrustFailureMode = TrustFailureMode.DENY,
        ttl: int = 300,
    ) -> None:
        """Initialize the Dominion Observatory verifier.

        Args:
            trust_threshold: Minimum `trust_score` required to pass
                verification. Scores strictly below this value raise
                `TrustVerificationError`.
            trust_failure_mode: Behavior when the trust API is unreachable.
                `ALLOW` permits execution; `DENY` raises `TrustVerificationError`.
            ttl: Time-to-live in seconds for cached trust scores. Subsequent
                calls for the same URL within this window skip the API request.
        """
        self.trust_threshold = trust_threshold
        self.trust_failure_mode = trust_failure_mode
        self.ttl = ttl
        # Maps server_url -> (TrustScore, expiry_monotonic_timestamp)
        self._cache: dict[str, tuple[TrustScore, float]] = {}

    # ------------------------------------------------------------------ #
    # Cache helpers
    # ------------------------------------------------------------------ #

    def _get_cached(self, server_url: str) -> TrustScore | None:
        entry = self._cache.get(server_url)
        if entry is None:
            return None
        score, expiry = entry
        if time.monotonic() < expiry:
            return score
        del self._cache[server_url]
        return None

    def _set_cached(self, server_url: str, score: TrustScore) -> None:
        self._cache[server_url] = (score, time.monotonic() + self.ttl)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _parse_response(self, data: dict[str, Any]) -> TrustScore:
        return TrustScore(
            trust_score=float(data["trust_score"]),
            sla_grade=str(data["sla_grade"]),
        )

    def _check_threshold(self, score: TrustScore, server_url: str) -> None:
        if score.trust_score < self.trust_threshold:
            msg = (
                f"MCP server '{server_url}' failed trust verification: "
                f"score {score.trust_score:.2f} is below threshold "
                f"{self.trust_threshold:.2f}"
            )
            raise TrustVerificationError(msg)

    def _unreachable_score(self, server_url: str) -> TrustScore:
        """Return a fallback score or raise, depending on trust_failure_mode."""
        if self.trust_failure_mode is TrustFailureMode.ALLOW:
            return TrustScore(trust_score=1.0, sla_grade="N/A")
        msg = (
            f"Trust API unreachable for '{server_url}'; "
            "blocking tool execution per DENY failure mode"
        )
        raise TrustVerificationError(msg)

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def verify(self, server_url: str) -> TrustScore:
        """Verify trust for `server_url` synchronously, using cache when available.

        Args:
            server_url: The URL of the MCP server to verify.

        Returns:
            `TrustScore` with the server's `trust_score` and `sla_grade`.

        Raises:
            TrustVerificationError: If `trust_score` is below `trust_threshold`,
                or if the API is unreachable and `trust_failure_mode` is `DENY`.
        """
        cached = self._get_cached(server_url)
        if cached is not None:
            self._check_threshold(cached, server_url)
            return cached

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(self._API_BASE, params={"url": server_url})
                response.raise_for_status()
                data: dict[str, Any] = response.json()
        except Exception:
            return self._unreachable_score(server_url)

        score = self._parse_response(data)
        self._set_cached(server_url, score)
        self._check_threshold(score, server_url)
        return score

    async def averify(self, server_url: str) -> TrustScore:
        """Verify trust for `server_url` asynchronously, using cache when available.

        Args:
            server_url: The URL of the MCP server to verify.

        Returns:
            `TrustScore` with the server's `trust_score` and `sla_grade`.

        Raises:
            TrustVerificationError: If `trust_score` is below `trust_threshold`,
                or if the API is unreachable and `trust_failure_mode` is `DENY`.
        """
        cached = self._get_cached(server_url)
        if cached is not None:
            self._check_threshold(cached, server_url)
            return cached

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(self._API_BASE, params={"url": server_url})
                response.raise_for_status()
                data: dict[str, Any] = response.json()
        except Exception:
            return self._unreachable_score(server_url)

        score = self._parse_response(data)
        self._set_cached(server_url, score)
        self._check_threshold(score, server_url)
        return score
