"""MCPToolkit — trust-verified gateway to Anthropic remote MCP servers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from langchain_anthropic.mcp.trust import (
    DominionObservatoryVerifier,
    TrustFailureMode,
    TrustScore,
    TrustVerificationError,
    TrustVerifier,
)

if TYPE_CHECKING:
    from langchain_anthropic.chat_models import ChatAnthropic


class _CallableTrustVerifier(TrustVerifier):
    """Adapts a plain callable into a TrustVerifier."""

    def __init__(self, fn: Callable[[str], TrustScore]) -> None:
        self._fn = fn

    def verify(self, server_url: str) -> TrustScore:
        return self._fn(server_url)

    async def averify(self, server_url: str) -> TrustScore:
        return self._fn(server_url)


class MCPToolkit:
    """Trust-verified toolkit for Anthropic remote MCP servers.

    `MCPToolkit` holds a set of MCP server configurations and verifies each
    server's trustworthiness before allowing a model to be constructed with
    those servers. Use `get_model()` / `aget_model()` to obtain a
    `ChatAnthropic` instance that is already bound to the verified servers.

    Example:
        .. code-block:: python

            from langchain_anthropic.mcp import MCPToolkit, TrustFailureMode

            toolkit = MCPToolkit(
                servers=[{"type": "url", "url": "https://mcp.example.com/mcp",
                          "name": "example-mcp"}],
                trust_threshold=0.8,
                trust_failure_mode=TrustFailureMode.DENY,
            )

            # Raises TrustVerificationError if any server fails verification
            model = toolkit.get_model(model="claude-sonnet-4-6")

    Args:
        servers: List of MCP server configuration dicts. Each dict must
            contain at least a ``"url"`` key with the server's endpoint.
            Additional keys (``"type"``, ``"name"``, ``"tool_configuration"``,
            etc.) are forwarded verbatim to the Anthropic API.
        trust_threshold: Minimum `trust_score` (0.0–1.0) required for a
            server to pass verification. Scores strictly below this value
            raise `TrustVerificationError`.
        trust_failure_mode: Behavior when the trust API is unreachable.
            `ALLOW` permits the model call; `DENY` raises
            `TrustVerificationError`.
        trust_verifier: Custom verifier to use instead of the default
            `DominionObservatoryVerifier`. Accepts either a `TrustVerifier`
            instance or a plain callable ``(server_url: str) -> TrustScore``.
            When ``None``, `DominionObservatoryVerifier` is constructed with
            the `trust_threshold`, `trust_failure_mode`, and `ttl` arguments.
        ttl: Cache TTL in seconds for trust scores. Ignored when a custom
            `trust_verifier` is provided.
    """

    def __init__(
        self,
        servers: list[dict[str, Any]],
        *,
        trust_threshold: float = 0.7,
        trust_failure_mode: TrustFailureMode = TrustFailureMode.DENY,
        trust_verifier: Callable[[str], TrustScore] | TrustVerifier | None = None,
        ttl: int = 300,
    ) -> None:
        self.servers = servers
        self.trust_threshold = trust_threshold
        self.trust_failure_mode = trust_failure_mode
        self.ttl = ttl
        self._verifier: TrustVerifier = self._resolve_verifier(trust_verifier)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _resolve_verifier(
        self,
        verifier: Callable[[str], TrustScore] | TrustVerifier | None,
    ) -> TrustVerifier:
        if verifier is None:
            return DominionObservatoryVerifier(
                trust_threshold=self.trust_threshold,
                trust_failure_mode=self.trust_failure_mode,
                ttl=self.ttl,
            )
        if isinstance(verifier, TrustVerifier):
            return verifier
        # Plain callable — wrap it
        return _CallableTrustVerifier(verifier)

    def _server_urls(self) -> list[str]:
        """Extract the URL from each server config dict."""
        return [s["url"] for s in self.servers if "url" in s]

    # ------------------------------------------------------------------ #
    # Trust verification
    # ------------------------------------------------------------------ #

    def verify_servers(self) -> list[TrustScore]:
        """Verify trust for all configured MCP servers synchronously.

        Returns:
            List of `TrustScore` objects, one per server URL.

        Raises:
            TrustVerificationError: If any server fails verification.
        """
        return [self._verifier.verify(url) for url in self._server_urls()]

    async def averify_servers(self) -> list[TrustScore]:
        """Verify trust for all configured MCP servers asynchronously.

        Returns:
            List of `TrustScore` objects, one per server URL.

        Raises:
            TrustVerificationError: If any server fails verification.
        """
        scores = []
        for url in self._server_urls():
            scores.append(await self._verifier.averify(url))
        return scores

    # ------------------------------------------------------------------ #
    # Model construction
    # ------------------------------------------------------------------ #

    def get_model(self, **kwargs: Any) -> ChatAnthropic:
        """Return a `ChatAnthropic` model pre-configured with verified MCP servers.

        Trust verification is performed synchronously before the model is
        constructed. If any server fails verification a `TrustVerificationError`
        is raised and no model is returned.

        Args:
            **kwargs: Forwarded verbatim to `ChatAnthropic.__init__`.

        Returns:
            A `ChatAnthropic` instance with `mcp_servers` set to the verified
            server list.

        Raises:
            TrustVerificationError: If any server fails trust verification.
        """
        from langchain_anthropic.chat_models import ChatAnthropic

        self.verify_servers()
        return ChatAnthropic(mcp_servers=self.servers, **kwargs)

    async def aget_model(self, **kwargs: Any) -> ChatAnthropic:
        """Return a `ChatAnthropic` model pre-configured with verified MCP servers.

        Trust verification is performed asynchronously before the model is
        constructed. If any server fails verification a `TrustVerificationError`
        is raised and no model is returned.

        Args:
            **kwargs: Forwarded verbatim to `ChatAnthropic.__init__`.

        Returns:
            A `ChatAnthropic` instance with `mcp_servers` set to the verified
            server list.

        Raises:
            TrustVerificationError: If any server fails trust verification.
        """
        from langchain_anthropic.chat_models import ChatAnthropic

        await self.averify_servers()
        return ChatAnthropic(mcp_servers=self.servers, **kwargs)


__all__ = [
    "MCPToolkit",
    "TrustVerificationError",
    "TrustFailureMode",
    "TrustScore",
    "TrustVerifier",
    "DominionObservatoryVerifier",
]
