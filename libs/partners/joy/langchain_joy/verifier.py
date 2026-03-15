"""Joy trust verification for LangChain agents."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import httpx

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

JOY_API_URL = "https://joy-connect.fly.dev"


class TrustVerificationError(Exception):
    """Raised when trust verification fails."""

    pass


@dataclass
class VerificationResult:
    """Result of agent trust verification."""

    is_trusted: bool
    agent_id: str
    trust_score: float
    vouch_count: int
    verified: bool
    capabilities: list[str]
    error: Optional[str] = None


class JoyTrustVerifier:
    """Verify agent trust using Joy network.

    Joy is a decentralized trust network where agents vouch for each other.
    This verifier checks an agent's trust score before allowing delegation.

    Example:
        verifier = JoyTrustVerifier(min_trust_score=0.5)

        # Simple check
        if verifier.should_trust("ag_xxx"):
            delegate_task(agent)

        # Detailed verification
        result = verifier.verify_agent("ag_xxx")
        print(f"Score: {result.trust_score}, Vouches: {result.vouch_count}")
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        min_trust_score: float = 0.5,
        require_verified: bool = False,
    ) -> None:
        """Initialize Joy verifier.

        Args:
            api_url: Joy API URL (default: from JOY_API_URL env or production)
            api_key: Joy API key (default: from JOY_API_KEY env)
            min_trust_score: Minimum trust score to consider agent trusted (0.0-2.0)
            require_verified: If True, only trust agents with verified badge
        """
        self.api_url = (
            api_url or os.getenv("JOY_API_URL") or JOY_API_URL
        ).rstrip("/")
        self.api_key = api_key or os.getenv("JOY_API_KEY")
        self.min_trust_score = min_trust_score
        self.require_verified = require_verified
        self._client: Optional[httpx.Client] = None

    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.Client(timeout=30.0)
        return self._client

    def _request(self, method: str, path: str, **kwargs: object) -> dict:
        """Make HTTP request to Joy API."""
        client = self._get_client()
        url = f"{self.api_url}{path}"

        headers = dict(kwargs.pop("headers", {}) or {})  # type: ignore[arg-type]
        headers["User-Agent"] = "langchain-joy/0.1.0"
        if self.api_key:
            headers["x-api-key"] = self.api_key

        response = client.request(method, url, headers=headers, **kwargs)  # type: ignore[arg-type]
        response.raise_for_status()
        return response.json()

    def verify_agent(self, agent_id: str) -> VerificationResult:
        """Verify an agent's trust status.

        Args:
            agent_id: Joy agent ID to verify (e.g., "ag_xxx")

        Returns:
            VerificationResult with trust details
        """
        try:
            data = self._request("GET", f"/agents/{agent_id}")

            trust_score = float(data.get("trust_score", 0))
            vouch_count = int(data.get("vouch_count", 0))
            verified = bool(data.get("verified", False))
            capabilities = data.get("capabilities", [])

            # Determine if trusted based on criteria
            is_trusted = trust_score >= self.min_trust_score
            if self.require_verified and not verified:
                is_trusted = False

            return VerificationResult(
                is_trusted=is_trusted,
                agent_id=agent_id,
                trust_score=trust_score,
                vouch_count=vouch_count,
                verified=verified,
                capabilities=capabilities,
            )

        except Exception as e:
            logger.exception("Trust verification failed for %s", agent_id)
            return VerificationResult(
                is_trusted=False,
                agent_id=agent_id,
                trust_score=0.0,
                vouch_count=0,
                verified=False,
                capabilities=[],
                error=str(e),
            )

    def should_trust(self, agent_id: str) -> bool:
        """Simple check if an agent should be trusted.

        Args:
            agent_id: Joy agent ID to check

        Returns:
            True if agent meets trust criteria
        """
        result = self.verify_agent(agent_id)
        return result.is_trusted

    def verify_before_delegation(
        self,
        agent_id: str,
        required_capabilities: Optional[list[str]] = None,
    ) -> bool:
        """Verify agent before delegating a task.

        Args:
            agent_id: Joy agent ID to verify
            required_capabilities: List of capabilities the agent must have

        Returns:
            True if agent is trusted and has required capabilities

        Raises:
            TrustVerificationError: If agent fails verification
        """
        result = self.verify_agent(agent_id)

        # Surface API errors
        if result.error:
            raise TrustVerificationError(
                f"Trust verification failed for {agent_id}: {result.error}"
            )

        if not result.is_trusted:
            raise TrustVerificationError(
                f"Agent {agent_id} not trusted: "
                f"score={result.trust_score} (min={self.min_trust_score})"
            )

        if required_capabilities:
            missing = set(required_capabilities) - set(result.capabilities)
            if missing:
                raise TrustVerificationError(
                    f"Agent {agent_id} missing capabilities: {missing}"
                )

        return True

    def discover_trusted_agents(
        self,
        capability: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 10,
    ) -> list[VerificationResult]:
        """Discover trusted agents from Joy network.

        Args:
            capability: Filter by capability (e.g., "code", "research")
            query: Search query
            limit: Maximum results

        Returns:
            List of VerificationResult for trusted agents
        """
        params: dict[str, object] = {"limit": limit}
        if capability:
            params["capability"] = capability
        if query:
            params["query"] = query

        try:
            data = self._request("GET", "/agents/discover", params=params)
            agents = data.get("agents", [])

            results = []
            for agent in agents:
                trust_score = float(agent.get("trust_score", 0))
                verified = bool(agent.get("verified", False))

                is_trusted = trust_score >= self.min_trust_score
                if self.require_verified and not verified:
                    is_trusted = False

                if is_trusted:
                    results.append(
                        VerificationResult(
                            is_trusted=True,
                            agent_id=agent.get("id", ""),
                            trust_score=trust_score,
                            vouch_count=int(agent.get("vouch_count", 0)),
                            verified=verified,
                            capabilities=agent.get("capabilities", []),
                        )
                    )

            return results

        except Exception as e:
            logger.exception("Agent discovery failed")
            return []

    def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            self._client.close()
            self._client = None

    def __enter__(self) -> "JoyTrustVerifier":
        """Context manager entry."""
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit."""
        self.close()
