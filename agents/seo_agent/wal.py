"""Write-Ahead Log (WAL) and Working Buffer for crash-resilient heartbeat execution.

The WAL ensures that heartbeat cycles are idempotent and crash-resilient.
Before executing any task, the heartbeat writes its plan to the WAL. As tasks
complete, they are marked done. On crash, the next cycle reads the incomplete
plan and resumes from where it left off.

The Working Buffer stores ephemeral state that must survive between heartbeat
cycles but isn't permanent enough for the main tables (e.g. "Ahrefs is
rate-limited today", "last cycle wrote a brief but didn't publish").

Usage::

    from agents.seo_agent.wal import WAL

    wal = WAL()
    cycle = wal.begin_cycle(planned_tasks=[
        {"task": "keyword_research", "site": "freeroomplanner"},
        {"task": "publish_blog", "site": "kitchen_estimator", "keyword": "kitchen cost UK"},
    ])

    for task in cycle.pending_tasks():
        cycle.mark_running(task["id"])
        try:
            result = run_task(task["task"], target_site=task["site"])
            cycle.mark_done(task["id"], result_summary=str(result)[:500])
        except Exception as e:
            cycle.mark_failed(task["id"], error=str(e))

    cycle.complete()
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task status constants
# ---------------------------------------------------------------------------

TASK_PLANNED = "planned"
TASK_RUNNING = "running"
TASK_DONE = "done"
TASK_FAILED = "failed"
TASK_SKIPPED = "skipped"

CYCLE_RUNNING = "running"
CYCLE_COMPLETED = "completed"
CYCLE_CRASHED = "crashed"
CYCLE_RESUMED = "resumed"


class WALCycle:
    """A single heartbeat cycle tracked in the WAL.

    Args:
        cycle_id: Unique cycle identifier.
        tasks: List of planned task dicts.
        resumed_from: ID of the crashed cycle this resumes, if any.
    """

    def __init__(
        self,
        cycle_id: str,
        tasks: list[dict[str, Any]],
        *,
        resumed_from: str | None = None,
    ) -> None:
        self.cycle_id = cycle_id
        self.tasks = tasks
        self.resumed_from = resumed_from
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.completed_at: str | None = None
        self.status = CYCLE_RESUMED if resumed_from else CYCLE_RUNNING
        self.notes: list[str] = []

    def pending_tasks(self) -> list[dict[str, Any]]:
        """Return tasks that haven't been completed or skipped."""
        return [
            t for t in self.tasks
            if t.get("status") in (TASK_PLANNED, TASK_RUNNING)
        ]

    def mark_running(self, task_id: str) -> None:
        """Mark a task as currently executing."""
        for t in self.tasks:
            if t["id"] == task_id:
                t["status"] = TASK_RUNNING
                t["started_at"] = datetime.now(timezone.utc).isoformat()
                break
        self._persist()

    def mark_done(self, task_id: str, *, result_summary: str = "") -> None:
        """Mark a task as successfully completed."""
        for t in self.tasks:
            if t["id"] == task_id:
                t["status"] = TASK_DONE
                t["completed_at"] = datetime.now(timezone.utc).isoformat()
                t["result_summary"] = result_summary[:500]
                break
        self._persist()

    def mark_failed(self, task_id: str, *, error: str = "") -> None:
        """Mark a task as failed."""
        for t in self.tasks:
            if t["id"] == task_id:
                t["status"] = TASK_FAILED
                t["completed_at"] = datetime.now(timezone.utc).isoformat()
                t["error"] = error[:500]
                break
        self._persist()

    def mark_skipped(self, task_id: str, *, reason: str = "") -> None:
        """Mark a task as intentionally skipped."""
        for t in self.tasks:
            if t["id"] == task_id:
                t["status"] = TASK_SKIPPED
                t["reason"] = reason[:200]
                break
        self._persist()

    def add_note(self, note: str) -> None:
        """Append a note to the cycle log (corrections, decisions, observations)."""
        self.notes.append(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {note}")
        self._persist()

    def complete(self) -> None:
        """Mark the cycle as successfully completed."""
        self.status = CYCLE_COMPLETED
        self.completed_at = datetime.now(timezone.utc).isoformat()
        self._persist()
        logger.info("WAL cycle %s completed", self.cycle_id)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the cycle to a dict for Supabase storage."""
        return {
            "cycle_id": self.cycle_id,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "resumed_from": self.resumed_from,
            "tasks": self.tasks,
            "notes": self.notes,
        }

    def _persist(self) -> None:
        """Write current state to Supabase."""
        try:
            from agents.seo_agent.tools.supabase_tools import upsert_record

            upsert_record(
                "heartbeat_wal",
                {
                    "cycle_id": self.cycle_id,
                    "status": self.status,
                    "started_at": self.started_at,
                    "completed_at": self.completed_at,
                    "resumed_from": self.resumed_from,
                    "tasks_json": self.tasks,
                    "notes_json": self.notes,
                },
                on_conflict="cycle_id",
            )
        except Exception:
            logger.warning("WAL persist failed (non-fatal)", exc_info=True)


class WorkingBuffer:
    """Ephemeral key-value state that survives between heartbeat cycles.

    Stores observations, partial results, and context that should carry
    forward to the next cycle but aren't permanent database records.

    Examples:
        - "ahrefs_rate_limited": True (skip Ahrefs calls this cycle)
        - "last_blog_keyword": "kitchen cost UK" (avoid repeating)
        - "consecutive_failures": 3 (trigger escalation)
    """

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        self._loaded = False

    def _load(self) -> None:
        """Load buffer from Supabase on first access."""
        if self._loaded:
            return
        self._loaded = True
        try:
            from agents.seo_agent.tools.supabase_tools import query_table

            rows = query_table("working_buffer", limit=200)
            for row in rows:
                self._cache[row["key"]] = {
                    "value": row.get("value_json"),
                    "expires_at": row.get("expires_at"),
                    "updated_at": row.get("updated_at"),
                }
        except Exception:
            logger.warning("Working buffer load failed (non-fatal)", exc_info=True)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the buffer.

        Args:
            key: The buffer key.
            default: Value to return if key is missing or expired.

        Returns:
            The stored value, or default.
        """
        self._load()
        entry = self._cache.get(key)
        if entry is None:
            return default

        # Check expiry
        expires = entry.get("expires_at")
        if expires:
            now = datetime.now(timezone.utc).isoformat()
            if now > expires:
                self.delete(key)
                return default

        return entry.get("value", default)

    def set(self, key: str, value: Any, *, ttl_hours: int = 24) -> None:
        """Store a value in the buffer.

        Args:
            key: The buffer key.
            value: Any JSON-serializable value.
            ttl_hours: Hours until the entry expires. Default 24h.
        """
        self._load()
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        expires_at = (now + timedelta(hours=ttl_hours)).isoformat()

        self._cache[key] = {
            "value": value,
            "expires_at": expires_at,
            "updated_at": now.isoformat(),
        }

        try:
            from agents.seo_agent.tools.supabase_tools import upsert_record

            upsert_record(
                "working_buffer",
                {
                    "key": key,
                    "value_json": value,
                    "expires_at": expires_at,
                    "updated_at": now.isoformat(),
                },
                on_conflict="key",
            )
        except Exception:
            logger.warning("Working buffer write failed for key=%s", key, exc_info=True)

    def delete(self, key: str) -> None:
        """Remove a key from the buffer."""
        self._cache.pop(key, None)
        try:
            from agents.seo_agent.tools.supabase_tools import get_client

            get_client().table("working_buffer").delete().eq("key", key).execute()
        except Exception:
            pass

    def increment(self, key: str, *, ttl_hours: int = 24) -> int:
        """Increment an integer counter in the buffer.

        Args:
            key: The counter key.
            ttl_hours: Hours until the counter expires.

        Returns:
            The new counter value.
        """
        current = self.get(key, 0)
        new_val = int(current) + 1
        self.set(key, new_val, ttl_hours=ttl_hours)
        return new_val

    def get_all(self) -> dict[str, Any]:
        """Return all non-expired buffer entries."""
        self._load()
        now = datetime.now(timezone.utc).isoformat()
        return {
            k: v["value"]
            for k, v in self._cache.items()
            if not v.get("expires_at") or v["expires_at"] > now
        }


class WAL:
    """Write-Ahead Log for heartbeat cycle management.

    Provides crash recovery by persisting the execution plan before running
    tasks, and tracking progress as tasks complete.
    """

    def __init__(self) -> None:
        self.buffer = WorkingBuffer()

    def begin_cycle(
        self,
        planned_tasks: list[dict[str, Any]],
    ) -> WALCycle:
        """Start a new heartbeat cycle.

        Checks for any crashed (incomplete) previous cycle first. If found,
        resumes it instead of starting fresh.

        Args:
            planned_tasks: List of task dicts with at least "task" and "site" keys.

        Returns:
            A ``WALCycle`` to track execution.
        """
        # Check for crashed cycles
        crashed = self._find_crashed_cycle()
        if crashed:
            logger.info(
                "Resuming crashed cycle %s (%d pending tasks)",
                crashed["cycle_id"],
                sum(1 for t in crashed.get("tasks_json", [])
                    if t.get("status") in (TASK_PLANNED, TASK_RUNNING)),
            )
            # Reset any "running" tasks back to "planned" (they were interrupted)
            tasks = crashed.get("tasks_json", [])
            for t in tasks:
                if t.get("status") == TASK_RUNNING:
                    t["status"] = TASK_PLANNED
                    t.pop("started_at", None)

            cycle = WALCycle(
                cycle_id=str(uuid.uuid4()),
                tasks=tasks,
                resumed_from=crashed["cycle_id"],
            )
            # Mark the old cycle as crashed
            try:
                from agents.seo_agent.tools.supabase_tools import upsert_record

                upsert_record(
                    "heartbeat_wal",
                    {
                        "cycle_id": crashed["cycle_id"],
                        "status": CYCLE_CRASHED,
                        "tasks_json": crashed.get("tasks_json", []),
                        "notes_json": crashed.get("notes_json", []),
                    },
                    on_conflict="cycle_id",
                )
            except Exception:
                pass

            cycle._persist()
            return cycle

        # No crashed cycle — start fresh
        tasks = []
        for t in planned_tasks:
            tasks.append({
                "id": str(uuid.uuid4()),
                "task": t["task"],
                "site": t.get("site", "all"),
                "params": {k: v for k, v in t.items() if k not in ("task", "site")},
                "status": TASK_PLANNED,
            })

        cycle = WALCycle(cycle_id=str(uuid.uuid4()), tasks=tasks)
        cycle._persist()
        logger.info("WAL cycle %s started with %d tasks", cycle.cycle_id, len(tasks))
        return cycle

    def get_recent_cycles(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent WAL cycles for inspection.

        Args:
            limit: Max cycles to return.

        Returns:
            List of cycle dicts, most recent first.
        """
        try:
            from agents.seo_agent.tools.supabase_tools import query_table

            return query_table(
                "heartbeat_wal",
                limit=limit,
                order_by="started_at",
                order_desc=True,
            )
        except Exception:
            return []

    def _find_crashed_cycle(self) -> dict[str, Any] | None:
        """Find the most recent cycle that was left in 'running' state."""
        try:
            from agents.seo_agent.tools.supabase_tools import query_table

            rows = query_table(
                "heartbeat_wal",
                filters={"status": CYCLE_RUNNING},
                limit=1,
                order_by="started_at",
                order_desc=True,
            )
            return rows[0] if rows else None
        except Exception:
            return None
