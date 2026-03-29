"""Gateway — persistent control plane for the Ralf agent.

The Gateway is the always-on process manager that coordinates all agent
subsystems. It provides:

1. **Service registry**: Tracks the health and status of all subsystems
   (Telegram bot, heartbeat worker, pulse checker).
2. **Unified execution layer**: All task execution goes through the gateway,
   which handles resource locking, rate limiting, and error isolation.
3. **Health monitoring**: Periodic self-checks with automatic recovery.
4. **File operations**: Read/write to GitHub repos (our "filesystem").
5. **Rate limit tracking**: Centralized tracking of API rate limits across
   all subsystems so they don't step on each other.

Unlike OpenClaw's local gateway (which runs on a user's machine with filesystem
access), Ralf's gateway is cloud-native — it uses GitHub API as its filesystem
and Supabase as its state store.

Usage::

    from agents.seo_agent.gateway import Gateway

    gw = Gateway()
    gw.boot()

    # Execute a task with resource locking and error isolation
    result = gw.execute_task("keyword_research", site="freeroomplanner")

    # Check system health
    health = gw.health_check()

    # Get execution context for a heartbeat cycle
    ctx = gw.get_execution_context()
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Service status tracking
# ---------------------------------------------------------------------------

SERVICE_HEALTHY = "healthy"
SERVICE_DEGRADED = "degraded"
SERVICE_DOWN = "down"
SERVICE_UNKNOWN = "unknown"


class ServiceStatus:
    """Health status of a single subsystem.

    Args:
        name: Service name.
        status: Current status.
        last_check: ISO timestamp of last health check.
        details: Optional details about the status.
    """

    def __init__(
        self,
        name: str,
        status: str = SERVICE_UNKNOWN,
        last_check: str | None = None,
        details: str = "",
    ) -> None:
        self.name = name
        self.status = status
        self.last_check = last_check or datetime.now(timezone.utc).isoformat()
        self.details = details
        self.consecutive_failures = 0

    def mark_healthy(self, details: str = "") -> None:
        """Mark service as healthy."""
        self.status = SERVICE_HEALTHY
        self.last_check = datetime.now(timezone.utc).isoformat()
        self.details = details
        self.consecutive_failures = 0

    def mark_degraded(self, details: str = "") -> None:
        """Mark service as degraded (partially working)."""
        self.status = SERVICE_DEGRADED
        self.last_check = datetime.now(timezone.utc).isoformat()
        self.details = details
        self.consecutive_failures += 1

    def mark_down(self, details: str = "") -> None:
        """Mark service as down."""
        self.status = SERVICE_DOWN
        self.last_check = datetime.now(timezone.utc).isoformat()
        self.details = details
        self.consecutive_failures += 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage/display."""
        return {
            "name": self.name,
            "status": self.status,
            "last_check": self.last_check,
            "details": self.details,
            "consecutive_failures": self.consecutive_failures,
        }


# ---------------------------------------------------------------------------
# Rate limit tracker
# ---------------------------------------------------------------------------


class RateLimitTracker:
    """Centralized rate limit tracking across all API integrations.

    Prevents multiple subsystems (heartbeat, Telegram bot, etc.) from
    simultaneously hitting rate-limited APIs.
    """

    def __init__(self, buffer: Any = None) -> None:
        self._buffer = buffer
        self._limits: dict[str, dict[str, Any]] = {}

    def record_limit(
        self,
        api: str,
        *,
        retry_after_seconds: int = 60,
        details: str = "",
    ) -> None:
        """Record that an API is rate-limited.

        Args:
            api: API identifier (e.g. "ahrefs", "openrouter", "github").
            retry_after_seconds: Seconds until the rate limit expires.
            details: Optional context.
        """
        expires_at = datetime.now(timezone.utc).timestamp() + retry_after_seconds
        self._limits[api] = {
            "expires_at": expires_at,
            "details": details,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.warning(
            "Rate limit recorded for %s: retry after %ds. %s",
            api, retry_after_seconds, details,
        )

        # Also store in working buffer for cross-process visibility
        if self._buffer:
            self._buffer.set(
                f"{api}_rate_limited",
                True,
                ttl_hours=max(1, retry_after_seconds // 3600 + 1),
            )

    def is_limited(self, api: str) -> bool:
        """Check if an API is currently rate-limited.

        Args:
            api: API identifier.

        Returns:
            True if the API is rate-limited.
        """
        limit = self._limits.get(api)
        if not limit:
            # Check buffer for cross-process limits
            if self._buffer and self._buffer.get(f"{api}_rate_limited"):
                return True
            return False

        if time.time() > limit["expires_at"]:
            del self._limits[api]
            if self._buffer:
                self._buffer.delete(f"{api}_rate_limited")
            return False
        return True

    def clear(self, api: str) -> None:
        """Clear a rate limit (e.g. after successful retry).

        Args:
            api: API identifier.
        """
        self._limits.pop(api, None)
        if self._buffer:
            self._buffer.delete(f"{api}_rate_limited")

    def active_limits(self) -> dict[str, dict[str, Any]]:
        """Return all currently active rate limits."""
        now = time.time()
        return {
            api: info for api, info in self._limits.items()
            if info["expires_at"] > now
        }


# ---------------------------------------------------------------------------
# Execution context
# ---------------------------------------------------------------------------


class ExecutionContext:
    """Context object passed to heartbeat cycles with all runtime state.

    Aggregates gateway services, budget, health, and buffer into a single
    object that task executors can query.

    Attributes:
        budget_remaining: Fraction of weekly LLM budget remaining (0.0-1.0).
        rate_limiter: Centralized rate limit tracker.
        services: Health status of all services.
        buffer: Working buffer for ephemeral state.
        memory: Episodic memory system.
        active_sites: Dict of currently active site profiles.
    """

    def __init__(
        self,
        *,
        budget_remaining: float,
        rate_limiter: RateLimitTracker,
        services: dict[str, ServiceStatus],
        buffer: Any,
        memory: Any,
        active_sites: dict[str, Any],
    ) -> None:
        self.budget_remaining = budget_remaining
        self.rate_limiter = rate_limiter
        self.services = services
        self.buffer = buffer
        self.memory = memory
        self.active_sites = active_sites

    def can_use_api(self, api: str) -> bool:
        """Check if an API can be used (not rate-limited and service healthy).

        Args:
            api: API identifier.

        Returns:
            True if the API is available.
        """
        if self.rate_limiter.is_limited(api):
            return False
        service = self.services.get(api)
        if service and service.status == SERVICE_DOWN:
            return False
        return True

    def can_spend(self, tier: str) -> bool:
        """Check if the budget allows spending at a given tier.

        Args:
            tier: Cost tier (haiku, sonnet, opus, none).

        Returns:
            True if the budget allows this spend.
        """
        if tier == "none":
            return True
        if self.budget_remaining < 0.05:
            return False
        if self.budget_remaining < 0.2 and tier in ("sonnet", "opus"):
            return False
        return True

    def summary(self) -> str:
        """Return a human-readable context summary."""
        lines = [
            f"Budget: {self.budget_remaining:.0%} remaining",
            f"Active sites: {', '.join(self.active_sites.keys())}",
        ]

        limits = self.rate_limiter.active_limits()
        if limits:
            lines.append(f"Rate limits: {', '.join(limits.keys())}")

        down = [s.name for s in self.services.values() if s.status == SERVICE_DOWN]
        if down:
            lines.append(f"Services down: {', '.join(down)}")

        return " | ".join(lines)


# ---------------------------------------------------------------------------
# Gateway
# ---------------------------------------------------------------------------


class Gateway:
    """Persistent control plane for the Ralf agent.

    Manages service health, rate limits, execution context, and provides
    the unified interface for all agent operations.
    """

    def __init__(self) -> None:
        from agents.seo_agent.wal import WorkingBuffer

        self.buffer = WorkingBuffer()
        self.rate_limiter = RateLimitTracker(buffer=self.buffer)
        self.services: dict[str, ServiceStatus] = {
            "supabase": ServiceStatus("supabase"),
            "openrouter": ServiceStatus("openrouter"),
            "ahrefs": ServiceStatus("ahrefs"),
            "github": ServiceStatus("github"),
            "telegram": ServiceStatus("telegram"),
            "gsc": ServiceStatus("gsc"),
        }
        self._booted = False

    def boot(self) -> dict[str, Any]:
        """Run startup health checks and initialize subsystems.

        Returns:
            Dict with boot status and any issues found.
        """
        issues: list[str] = []

        # Check Supabase
        try:
            from agents.seo_agent.tools.supabase_tools import ensure_tables, get_client

            get_client()
            ensure_tables()
            self.services["supabase"].mark_healthy()
        except Exception as e:
            self.services["supabase"].mark_down(str(e)[:200])
            issues.append(f"Supabase: {e}")

        # Check OpenRouter
        if os.environ.get("OPENROUTER_API_KEY"):
            self.services["openrouter"].mark_healthy("API key present")
        elif os.environ.get("ANTHROPIC_API_KEY"):
            self.services["openrouter"].mark_degraded("Using Anthropic fallback")
        else:
            self.services["openrouter"].mark_down("No LLM API key")
            issues.append("No LLM API key configured")

        # Check Ahrefs
        if os.environ.get("AHREFS_API_KEY"):
            self.services["ahrefs"].mark_healthy("API key present")
        else:
            self.services["ahrefs"].mark_degraded("No API key — keyword research unavailable")

        # Check GitHub
        if os.environ.get("GITHUB_TOKEN"):
            self.services["github"].mark_healthy("Token present")
        else:
            self.services["github"].mark_degraded("No token — blog publishing unavailable")

        # Check Telegram
        if os.environ.get("TELEGRAM_BOT_TOKEN"):
            self.services["telegram"].mark_healthy("Token present")
        else:
            self.services["telegram"].mark_down("No bot token")
            issues.append("No Telegram bot token")

        # Check GSC
        from agents.seo_agent.tools.gsc_tools import test_connection as gsc_test_connection

        gsc_result = gsc_test_connection()
        if gsc_result["ok"]:
            self.services["gsc"].mark_healthy(gsc_result["detail"])
        else:
            self.services["gsc"].mark_degraded(gsc_result["detail"])

        self._booted = True
        logger.info(
            "Gateway booted: %d healthy, %d degraded, %d down",
            sum(1 for s in self.services.values() if s.status == SERVICE_HEALTHY),
            sum(1 for s in self.services.values() if s.status == SERVICE_DEGRADED),
            sum(1 for s in self.services.values() if s.status == SERVICE_DOWN),
        )

        return {
            "booted": True,
            "issues": issues,
            "services": {k: v.to_dict() for k, v in self.services.items()},
        }

    def health_check(self) -> dict[str, Any]:
        """Run a quick health check on all services.

        Returns:
            Dict with service statuses and overall health.
        """
        if not self._booted:
            self.boot()

        # Re-check Supabase connectivity
        try:
            from agents.seo_agent.tools.supabase_tools import get_weekly_spend

            spend = get_weekly_spend()
            self.services["supabase"].mark_healthy(f"Connected, spend=${spend:.4f}")
        except Exception as e:
            self.services["supabase"].mark_degraded(str(e)[:100])

        overall = SERVICE_HEALTHY
        for s in self.services.values():
            if s.status == SERVICE_DOWN:
                overall = SERVICE_DOWN
                break
            if s.status == SERVICE_DEGRADED:
                overall = SERVICE_DEGRADED

        return {
            "overall": overall,
            "services": {k: v.to_dict() for k, v in self.services.items()},
            "rate_limits": self.rate_limiter.active_limits(),
            "buffer_entries": len(self.buffer.get_all()),
        }

    def get_execution_context(self) -> ExecutionContext:
        """Build a full execution context for a heartbeat cycle.

        Returns:
            An ``ExecutionContext`` with budget, services, buffer, memory, and sites.
        """
        if not self._booted:
            self.boot()

        from agents.seo_agent.config import MAX_WEEKLY_SPEND_USD, SITE_PROFILES
        from agents.seo_agent.memory import Memory
        from agents.seo_agent.tools.supabase_tools import get_weekly_spend

        spend = get_weekly_spend()
        cap = float(os.getenv("MAX_WEEKLY_SPEND_USD", str(MAX_WEEKLY_SPEND_USD)))
        budget_remaining = max(0.0, 1.0 - (spend / cap)) if cap > 0 else 0.0

        active_sites = {
            k: v for k, v in SITE_PROFILES.items()
            if v.get("status") == "active"
        }

        return ExecutionContext(
            budget_remaining=budget_remaining,
            rate_limiter=self.rate_limiter,
            services=self.services,
            buffer=self.buffer,
            memory=Memory(),
            active_sites=active_sites,
        )

    def execute_task(
        self,
        task_type: str,
        *,
        site: str = "all",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute a task with error isolation and rate limit handling.

        Wraps the LangGraph task execution with gateway-level concerns:
        pre-checks, error capture, rate limit detection, and post-execution
        memory updates.

        Args:
            task_type: The task to run.
            site: Target site key.
            **kwargs: Additional task parameters.

        Returns:
            The task result dict.
        """
        from agents.seo_agent.agent import build_graph, create_initial_state
        from agents.seo_agent.tools.supabase_tools import ensure_tables, get_weekly_spend

        # Pre-check: is the required API available?
        api_map = {
            "keyword_research": "ahrefs",
            "content_gap": "ahrefs",
            "discover_prospects": "ahrefs",
            "rank_tracker": "gsc",
        }
        required_api = api_map.get(task_type)
        if required_api and self.rate_limiter.is_limited(required_api):
            logger.warning("Skipping %s: %s is rate-limited", task_type, required_api)
            return {"errors": [f"{required_api} is rate-limited"], "skipped": True}

        ensure_tables()
        weekly_spend = get_weekly_spend()

        state = create_initial_state(task_type=task_type, target_site=site, **kwargs)
        state["llm_spend_this_week"] = weekly_spend

        try:
            graph = build_graph()
            result = graph.invoke(state)

            # Post-execution: update service health
            if required_api:
                self.services.get(required_api, ServiceStatus(required_api)).mark_healthy()

            return result

        except Exception as e:
            error_str = str(e).lower()

            # Detect rate limits from error messages
            if "rate limit" in error_str or "429" in error_str or "too many requests" in error_str:
                api = required_api or "openrouter"
                self.rate_limiter.record_limit(api, retry_after_seconds=300, details=str(e)[:200])
                if required_api:
                    self.services[required_api].mark_degraded(f"Rate limited: {e}")

            # Update service health on failure
            if required_api:
                svc = self.services.get(required_api)
                if svc:
                    svc.mark_degraded(str(e)[:200])

            raise

    def format_status_report(self) -> str:
        """Format a concise status report for Telegram.

        Returns:
            Human-readable status string.
        """
        health = self.health_check()
        lines = [f"System: {health['overall'].upper()}"]

        for name, info in health["services"].items():
            icon = {"healthy": "+", "degraded": "~", "down": "X", "unknown": "?"}.get(
                info["status"], "?"
            )
            lines.append(f"  [{icon}] {name}: {info['details'][:60]}" if info["details"] else f"  [{icon}] {name}")

        limits = health.get("rate_limits", {})
        if limits:
            lines.append(f"Rate limits: {', '.join(limits.keys())}")

        return "\n".join(lines)
