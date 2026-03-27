"""Lightweight health and status API for the Ralf agent.

Runs a minimal HTTP server alongside the Telegram bot, providing:
- ``GET /health`` — Railway health check endpoint
- ``GET /status`` — Full system status (services, rate limits, budget)
- ``GET /trigger/<task>`` — Manually trigger a cron job

Uses Python's built-in ``http.server`` to avoid adding FastAPI/Flask
as a dependency. Designed to run in a background thread.

Usage::

    from agents.seo_agent.api import start_health_server

    # Start on port 18789 (non-blocking, runs in a daemon thread)
    start_health_server(port=18789)
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

logger = logging.getLogger(__name__)

# When the server booted (for uptime reporting)
_BOOT_TIME: str = datetime.now(timezone.utc).isoformat()


class HealthHandler(BaseHTTPRequestHandler):
    """HTTP request handler for health and status endpoints."""

    def do_GET(self) -> None:
        """Route GET requests to the appropriate handler."""
        path = self.path.rstrip("/")

        if path == "/health":
            self._handle_health()
        elif path == "/status":
            self._handle_status()
        elif path.startswith("/trigger/"):
            task = path.split("/trigger/", 1)[1]
            self._handle_trigger(task)
        else:
            self._send_json(404, {"error": "not_found", "endpoints": ["/health", "/status", "/trigger/<task>"]})

    def _handle_health(self) -> None:
        """Minimal health check for Railway/Docker monitoring."""
        self._send_json(200, {
            "status": "ok",
            "booted_at": _BOOT_TIME,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def _handle_status(self) -> None:
        """Full system status with services, rate limits, and budget."""
        try:
            from agents.seo_agent.gateway import Gateway

            gw = Gateway()
            health = gw.health_check()

            # Add budget info
            from agents.seo_agent.tools.supabase_tools import get_weekly_spend

            spend = get_weekly_spend()
            cap = float(os.getenv("MAX_WEEKLY_SPEND_USD", "50.00"))

            self._send_json(200, {
                "status": "ok",
                "booted_at": _BOOT_TIME,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "gateway": health,
                "budget": {
                    "spent": round(spend, 4),
                    "cap": cap,
                    "remaining_pct": round(max(0, 1 - spend / cap) * 100, 1) if cap > 0 else 0,
                },
            })
        except Exception as e:
            self._send_json(500, {"status": "error", "error": str(e)[:300]})

    def _handle_trigger(self, task: str) -> None:
        """Manually trigger a worker or pulse cycle."""
        valid_tasks = {"worker", "pulse", "heartbeat"}
        if task not in valid_tasks:
            self._send_json(400, {
                "error": "invalid_task",
                "valid_tasks": sorted(valid_tasks),
            })
            return

        # Fire the task in a background thread to avoid blocking the API
        def _run() -> None:
            import asyncio

            try:
                if task == "worker":
                    from agents.seo_agent.worker import execute_worker_cycle
                    asyncio.run(execute_worker_cycle())
                elif task == "pulse":
                    from agents.seo_agent.pulse import execute_pulse
                    asyncio.run(execute_pulse())
                elif task == "heartbeat":
                    from agents.seo_agent.heartbeat import execute_heartbeat
                    asyncio.run(execute_heartbeat())
            except Exception:
                logger.error("Triggered %s failed", task, exc_info=True)

        thread = threading.Thread(target=_run, daemon=True, name=f"trigger-{task}")
        thread.start()

        self._send_json(202, {
            "status": "accepted",
            "task": task,
            "message": f"{task} triggered in background",
        })

    def _send_json(self, status_code: int, data: dict[str, Any]) -> None:
        """Send a JSON response."""
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default access logs to avoid noise; log via our logger."""
        logger.debug("API %s", format % args)


def start_health_server(*, port: int | None = None) -> HTTPServer | None:
    """Start the health server in a background daemon thread.

    Args:
        port: Port to listen on. Defaults to ``GATEWAY_PORT`` env var or 18789.

    Returns:
        The ``HTTPServer`` instance, or ``None`` if startup failed.
    """
    port = port or int(os.getenv("GATEWAY_PORT", "18789"))

    try:
        server = HTTPServer(("0.0.0.0", port), HealthHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True, name="health-api")
        thread.start()
        logger.info("Health API running on port %d", port)
        return server
    except OSError as e:
        logger.warning("Could not start health API on port %d: %s", port, e)
        return None
