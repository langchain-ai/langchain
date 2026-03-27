"""Cron execution tracking and deduplication for scheduled agent tasks.

Prevents overlapping executions and logs every cron run to Supabase for
audit history. The Telegram bot's ``job_queue`` calls ``acquire_lock``
before running a worker/pulse cycle and ``release_lock`` when done.

Usage::

    from agents.seo_agent.cron import CronTracker

    tracker = CronTracker()

    if tracker.acquire_lock("worker"):
        try:
            result = await execute_worker_cycle()
            tracker.release_lock("worker", status="completed", tasks_executed=3)
        except Exception as e:
            tracker.release_lock("worker", status="failed", error=str(e))
    else:
        logger.info("Worker already running — skipping this cycle")
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Default schedule config — can be overridden by cron.json
DEFAULT_SCHEDULE: dict[str, dict[str, Any]] = {
    "worker": {
        "interval_hours": int(os.getenv("WORKER_INTERVAL_HOURS", "3")),
        "description": "Heavy background tasks (content writing, keyword research, prospecting)",
        "first_delay_seconds": 600,
    },
    "pulse": {
        "interval_minutes": int(os.getenv("PULSE_INTERVAL_MINUTES", "60")),
        "description": "Lightweight check-in (ranking movers, budget alerts, progress)",
        "first_delay_seconds": 300,
    },
}


def load_schedule() -> dict[str, dict[str, Any]]:
    """Load schedule from cron.json if it exists, otherwise use defaults.

    Returns:
        Dict of job configs keyed by job_id.
    """
    cron_path = os.path.join(os.path.dirname(__file__), "..", "..", "cron.json")
    cron_path = os.path.normpath(cron_path)

    if os.path.exists(cron_path):
        try:
            with open(cron_path) as f:
                data = json.load(f)
            jobs = {j["id"]: j for j in data.get("jobs", [])}
            logger.info("Loaded %d jobs from cron.json", len(jobs))
            return jobs
        except Exception:
            logger.warning("Failed to parse cron.json, using defaults", exc_info=True)

    return DEFAULT_SCHEDULE


class CronTracker:
    """Tracks cron execution and prevents overlapping runs.

    Uses Supabase's ``cron_executions`` table for persistent state,
    with an in-memory fallback if the table doesn't exist yet.
    """

    def __init__(self) -> None:
        self._locks: dict[str, str] = {}  # job_id -> execution_id (in-memory fallback)

    def acquire_lock(self, job_id: str, *, max_age_minutes: int = 120) -> bool:
        """Try to acquire an execution lock for a job.

        Checks if there's already a running execution for this job that
        started less than ``max_age_minutes`` ago. If so, returns False
        (skip this cycle). Otherwise, creates a new execution record and
        returns True.

        Args:
            job_id: The cron job identifier (e.g. "worker", "pulse").
            max_age_minutes: Consider runs older than this as stale/crashed.

        Returns:
            True if the lock was acquired (safe to proceed).
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)).isoformat()

        try:
            from agents.seo_agent.tools.supabase_tools import get_client, insert_record

            client = get_client()

            # Check for running executions within the window
            result = (
                client.table("cron_executions")
                .select("id, fired_at")
                .eq("job_id", job_id)
                .eq("status", "running")
                .gte("fired_at", cutoff)
                .execute()
            )

            if result.data:
                running = result.data[0]
                logger.info(
                    "Skipping %s: already running since %s (id=%s)",
                    job_id, running["fired_at"], running["id"],
                )
                return False

            # Create new execution record
            record = insert_record("cron_executions", {
                "job_id": job_id,
                "fired_at": datetime.now(timezone.utc).isoformat(),
                "status": "running",
            })
            exec_id = record.get("id", "unknown")
            self._locks[job_id] = exec_id
            logger.info("Acquired lock for %s (execution_id=%s)", job_id, exec_id)
            return True

        except Exception:
            # Supabase table might not exist yet — fall back to in-memory
            logger.debug("Cron lock via Supabase failed, using in-memory fallback", exc_info=True)
            if job_id in self._locks:
                logger.info("Skipping %s: in-memory lock held", job_id)
                return False
            self._locks[job_id] = "in-memory"
            return True

    def release_lock(
        self,
        job_id: str,
        *,
        status: str = "completed",
        tasks_executed: int = 0,
        tokens_used: int = 0,
        message_sent: bool = False,
        error: str = "",
    ) -> None:
        """Release the execution lock and update the record.

        Args:
            job_id: The cron job identifier.
            status: Final status ("completed" or "failed").
            tasks_executed: Number of tasks completed this cycle.
            tokens_used: Total tokens consumed.
            message_sent: Whether a Telegram message was sent.
            error: Error message if the job failed.
        """
        exec_id = self._locks.pop(job_id, None)

        if not exec_id or exec_id == "in-memory":
            return

        try:
            from agents.seo_agent.tools.supabase_tools import get_client

            client = get_client()
            client.table("cron_executions").update({
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "status": status,
                "tasks_executed": tasks_executed,
                "tokens_used": tokens_used,
                "message_sent": message_sent,
                "error": error[:500] if error else None,
            }).eq("id", exec_id).execute()

            logger.info("Released lock for %s: %s", job_id, status)
        except Exception:
            logger.debug("Cron lock release failed (non-fatal)", exc_info=True)

    def recent_executions(self, job_id: str, *, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent execution history for a job.

        Args:
            job_id: The cron job identifier.
            limit: Max records to return.

        Returns:
            List of execution records, newest first.
        """
        try:
            from agents.seo_agent.tools.supabase_tools import get_client

            client = get_client()
            result = (
                client.table("cron_executions")
                .select("*")
                .eq("job_id", job_id)
                .order("fired_at", desc=True)
                .limit(limit)
                .execute()
            )
            return result.data or []
        except Exception:
            return []
