"""Joy Trust Network API client."""

from __future__ import annotations

import time
from typing import Any

import httpx


class JoyTrustError(Exception):
    """Error from Joy Trust API."""

    pass


class JoyTrustClient:
    """Client for Joy Trust Network API.

    Provides methods to query agent trust scores, discover agents,
    and verify trust thresholds.

    Example:
        >>> client = JoyTrustClient()
        >>> result = client.get_trust_score("ag_abc123")
        >>> print(result["trust_score"])
        2.3
    """

    DEFAULT_BASE_URL = "https://choosejoy.com.au"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 10.0,
        cache_ttl: int = 300,
    ) -> None:
        """Initialize Joy Trust client.

        Args:
            api_key: Optional API key for higher rate limits.
            base_url: Override base URL (default: https://choosejoy.com.au).
            timeout: Request timeout in seconds.
            cache_ttl: Cache TTL in seconds (default: 300).
        """
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self._cache: dict[str, tuple[float, Any]] = {}

    def _get_headers(self) -> dict[str, str]:
        """Get request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def _get_cached(self, key: str) -> Any | None:
        """Get cached value if not expired."""
        if key in self._cache:
            timestamp, value = self._cache[key]
            if time.time() - timestamp < self.cache_ttl:
                return value
            del self._cache[key]
        return None

    def _set_cached(self, key: str, value: Any) -> None:
        """Set cached value."""
        self._cache[key] = (time.time(), value)

    def get_trust_score(self, agent_id: str) -> dict[str, Any]:
        """Get trust score for an agent.

        Args:
            agent_id: The agent ID to look up.

        Returns:
            Dict with trust_score, verified, vouch_count, etc.

        Raises:
            JoyTrustError: If API request fails.
        """
        cache_key = f"trust:{agent_id}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(
                    f"{self.base_url}/agents/{agent_id}",
                    headers=self._get_headers(),
                )
                response.raise_for_status()
                data = response.json()
                self._set_cached(cache_key, data)
                return data
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return {"agent_id": agent_id, "trust_score": 0.0, "found": False}
            raise JoyTrustError(f"API error: {e.response.status_code}") from e
        except httpx.RequestError as e:
            raise JoyTrustError(f"Request failed: {e}") from e

    def verify_trust(
        self,
        agent_id: str,
        *,
        min_trust: float = 1.5,
    ) -> dict[str, Any]:
        """Verify if agent meets minimum trust threshold.

        Args:
            agent_id: The agent ID to verify.
            min_trust: Minimum trust score required.

        Returns:
            Dict with meets_threshold, trust_score, etc.
        """
        result = self.get_trust_score(agent_id)
        trust_score = result.get("trust_score", 0.0)
        return {
            "agent_id": agent_id,
            "trust_score": trust_score,
            "threshold": min_trust,
            "meets_threshold": trust_score >= min_trust,
            "verified": result.get("verified", False),
        }

    def discover_agents(
        self,
        *,
        query: str | None = None,
        capability: str | None = None,
        min_trust: float | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Discover agents by capability or search query.

        Args:
            query: Free-text search query.
            capability: Filter by capability.
            min_trust: Minimum trust score filter.
            limit: Maximum results to return.

        Returns:
            List of agent dictionaries.
        """
        params: dict[str, Any] = {"limit": limit}
        if query:
            params["query"] = query
        if capability:
            params["capability"] = capability

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(
                    f"{self.base_url}/agents/discover",
                    params=params,
                    headers=self._get_headers(),
                )
                response.raise_for_status()
                data = response.json()
                agents = data.get("agents", [])

                # Filter by min_trust if specified
                if min_trust is not None:
                    agents = [
                        a for a in agents if a.get("trust_score", 0) >= min_trust
                    ]

                return agents
        except httpx.RequestError as e:
            raise JoyTrustError(f"Discovery failed: {e}") from e
