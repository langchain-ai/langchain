"""Persistent episodic memory for Ralf.

Unlike the static MEMORY.md, this module provides a living memory system that
evolves with every interaction and heartbeat cycle. Memories are stored in
Supabase and recalled contextually.

Memory categories:
    - ``user_preference``: Things the user likes/dislikes (e.g. short updates)
    - ``learning``: SEO insights learned from data (e.g. "room planner keywords outperform kitchen")
    - ``performance``: Observations about what worked vs. what didn't
    - ``correction``: Mistakes acknowledged and behaviour changes committed to
    - ``decision``: Strategic decisions made (e.g. "pausing outreach until DR > 10")
    - ``context``: Ongoing situational context (e.g. "Ahrefs API flaky this week")
    - ``routine``: Recurring patterns (e.g. "Ben checks in on Mondays around 9am")

Usage::

    from agents.seo_agent.memory import Memory

    mem = Memory()

    # Store a new memory
    mem.store("learning", "Posts about room planning get 3x more clicks than kitchen posts")

    # Recall relevant memories for a prompt
    context = mem.recall_for_prompt(topic="content strategy")
    # Returns formatted string of relevant memories

    # Auto-learn from heartbeat outcomes
    mem.learn_from_outcome(task="publish_blog", keyword="room planner guide", success=True,
                           metrics={"clicks_7d": 45})
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

MEMORY_CATEGORIES = [
    "user_preference",
    "learning",
    "performance",
    "correction",
    "decision",
    "context",
    "routine",
    "activity",
]

# How many memories to include in prompts (to avoid blowing up context)
MAX_MEMORIES_IN_PROMPT = 15

# Memories older than this are candidates for consolidation
CONSOLIDATION_THRESHOLD_DAYS = 30


class Memory:
    """Persistent episodic memory backed by Supabase.

    Stores categorised memories with relevance scoring and automatic
    consolidation of old entries.
    """

    def __init__(self) -> None:
        self._cache: list[dict[str, Any]] | None = None

    def store(
        self,
        category: str,
        content: str,
        *,
        importance: int = 5,
        source: str = "heartbeat",
        related_site: str = "",
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Store a new memory.

        Args:
            category: One of ``MEMORY_CATEGORIES``.
            content: The memory content (keep concise, 1-2 sentences).
            importance: 1-10 scale, where 10 is critical. Default 5.
            source: Where this memory came from (heartbeat, telegram, etc).
            related_site: Site key if site-specific.
            tags: Optional tags for retrieval.

        Returns:
            The stored memory record.
        """
        from agents.seo_agent.tools.supabase_tools import insert_record

        if category not in MEMORY_CATEGORIES:
            logger.warning("Unknown memory category: %s", category)

        record = {
            "category": category,
            "content": content[:500],
            "importance": max(1, min(10, importance)),
            "source": source,
            "related_site": related_site,
            "tags": tags or [],
            "recall_count": 0,
            "superseded_by": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            result = insert_record("ralf_memory", record)
            self._cache = None  # Invalidate cache
            logger.info("Stored memory [%s]: %s", category, content[:80])
            return result
        except Exception:
            logger.warning("Memory store failed", exc_info=True)
            return record

    def recall(
        self,
        *,
        category: str | None = None,
        related_site: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Recall memories, optionally filtered.

        Args:
            category: Filter by category.
            related_site: Filter by site.
            limit: Max memories to return.

        Returns:
            List of memory dicts, ordered by importance then recency.
        """
        from agents.seo_agent.tools.supabase_tools import query_table

        filters: dict[str, Any] = {}
        if category:
            filters["category"] = category
        if related_site:
            filters["related_site"] = related_site

        try:
            rows = query_table(
                "ralf_memory",
                filters=filters if filters else None,
                limit=limit,
                order_by="importance",
                order_desc=True,
            )
            # Filter out superseded memories
            return [r for r in rows if not r.get("superseded_by")]
        except Exception:
            logger.warning("Memory recall failed", exc_info=True)
            return []

    def recall_for_prompt(
        self,
        *,
        topic: str = "",
        site: str = "",
        max_entries: int = MAX_MEMORIES_IN_PROMPT,
    ) -> str:
        """Build a memory context string suitable for injection into LLM prompts.

        Retrieves the most important and relevant memories and formats them
        as a concise context block.

        Args:
            topic: Optional topic to bias recall towards.
            site: Optional site to filter by.
            max_entries: Max memories to include.

        Returns:
            Formatted string of memories, or empty string if none.
        """
        memories = self._get_all_active()

        # Score and rank memories
        scored: list[tuple[float, dict]] = []
        for m in memories:
            score = float(m.get("importance", 5))

            # Boost if site matches
            if site and m.get("related_site") == site:
                score += 2.0

            # Boost if topic overlaps with content or tags
            if topic:
                topic_words = set(topic.lower().split())
                content_words = set(m.get("content", "").lower().split())
                tag_words = set(t.lower() for t in m.get("tags", []))
                overlap = topic_words & (content_words | tag_words)
                score += len(overlap) * 1.5

            # Boost corrections and user preferences (always relevant)
            if m.get("category") in ("correction", "user_preference"):
                score += 2.0

            # Boost activity memories when topic suggests a "what have you done?" query
            _activity_signals = {
                "done", "written", "published", "did", "activities",
                "history", "recent", "blog", "posts",
            }
            if topic and m.get("category") == "activity":
                if topic_words & _activity_signals:
                    score += 3.0

            scored.append((score, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:max_entries]

        if not top:
            return ""

        # Increment recall counts
        for _, m in top:
            self._increment_recall(m.get("id", ""))

        # Format for prompt
        lines = ["\n--- Ralf's Memory ---"]
        for _, m in top:
            cat = m.get("category", "")
            content = m.get("content", "")
            site_tag = f" [{m['related_site']}]" if m.get("related_site") else ""
            lines.append(f"[{cat}]{site_tag} {content}")
        lines.append("--- End Memory ---\n")

        return "\n".join(lines)

    def learn_from_outcome(
        self,
        *,
        task: str,
        success: bool,
        details: str = "",
        site: str = "",
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Automatically derive a memory from a task outcome.

        Called by the heartbeat after each task to build up pattern recognition.

        Args:
            task: Task type that was executed.
            success: Whether the task succeeded.
            details: Human-readable description of what happened.
            site: Related site key.
            metrics: Optional performance metrics.
        """
        if success and metrics:
            # Look for notable performance
            clicks = metrics.get("clicks_7d", 0)
            if clicks > 20:
                self.store(
                    "performance",
                    f"Task '{task}' for {site} performing well: {details}. "
                    f"Metrics: {metrics}",
                    importance=7,
                    source="auto_learn",
                    related_site=site,
                    tags=[task, "high_performance"],
                )
        elif not success:
            # Record failures for pattern detection
            self.store(
                "context",
                f"Task '{task}' failed for {site}: {details[:200]}",
                importance=4,
                source="auto_learn",
                related_site=site,
                tags=[task, "failure"],
            )

            # Check for repeated failures
            recent_failures = self.recall(category="context", limit=10)
            task_failures = [
                m for m in recent_failures
                if task in m.get("content", "") and "failed" in m.get("content", "")
            ]
            if len(task_failures) >= 3:
                self.store(
                    "learning",
                    f"Task '{task}' has failed {len(task_failures)} times recently. "
                    f"May need investigation or temporary skip.",
                    importance=8,
                    source="auto_learn",
                    related_site=site,
                    tags=[task, "repeated_failure", "escalate"],
                )

    def store_user_preference(self, preference: str, *, source: str = "telegram") -> None:
        """Store a user preference learned from conversation.

        Args:
            preference: The preference (e.g. "prefers short updates").
            source: Where this was learned.
        """
        # Check for duplicate preferences
        existing = self.recall(category="user_preference", limit=50)
        for m in existing:
            if _similar_content(m.get("content", ""), preference):
                logger.info("Preference already stored: %s", preference[:50])
                return

        self.store(
            "user_preference",
            preference,
            importance=8,
            source=source,
            tags=["preference"],
        )

    def log_activity(
        self,
        action_type: str,
        summary: str,
        *,
        site: str = "",
        details: dict[str, Any] | None = None,
        source: str = "worker",
    ) -> dict[str, Any]:
        """Log a completed activity for later recall.

        Activities record what the agent did, enabling it to answer questions
        like "what blog posts have you written?"

        Args:
            action_type: e.g. "blog_published", "keyword_research".
            summary: Human-readable one-liner of what happened.
            site: Related site key.
            details: Optional structured data (title, URL, keyword, etc.).
            source: Where this activity originated.

        Returns:
            The stored memory record.
        """
        content = summary
        if details:
            detail_parts = [f"{k}={v}" for k, v in details.items() if v]
            if detail_parts:
                content += f" | {', '.join(detail_parts[:5])}"

        tags = [action_type]
        if site:
            tags.append(site)

        return self.store(
            "activity",
            content[:500],
            importance=3,
            source=source,
            related_site=site,
            tags=tags,
        )

    def recall_activities(
        self,
        *,
        action_type: str | None = None,
        site: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Recall past activities, optionally filtered by type and site.

        Args:
            action_type: Filter to a specific action (e.g. "blog_published").
            site: Filter to a specific site.
            limit: Max results.

        Returns:
            List of activity memory dicts, most recent first.
        """
        from agents.seo_agent.tools.supabase_tools import query_table

        filters: dict[str, Any] = {"category": "activity"}
        if site:
            filters["related_site"] = site

        try:
            rows = query_table(
                "ralf_memory",
                filters=filters,
                limit=limit,
                order_by="created_at",
                order_desc=True,
            )
            if action_type:
                rows = [r for r in rows if action_type in r.get("tags", [])]
            return [r for r in rows if not r.get("superseded_by")]
        except Exception:
            logger.warning("Activity recall failed", exc_info=True)
            return []

    def store_correction(self, correction: str, *, source: str = "telegram") -> None:
        """Store a behavioural correction from the user.

        These are high-importance memories that should always surface in prompts.

        Args:
            correction: What the user corrected (e.g. "stop writing about B&Q").
            source: Where this correction came from.
        """
        self.store(
            "correction",
            correction,
            importance=9,
            source=source,
            tags=["correction", "behaviour_change"],
        )

    def consolidate(self) -> int:
        """Consolidate old, low-importance memories to keep the store manageable.

        Merges similar old memories and removes superseded ones.

        Returns:
            Number of memories consolidated.
        """
        from datetime import timedelta

        memories = self._get_all_active()
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=CONSOLIDATION_THRESHOLD_DAYS)
        ).isoformat()

        old_memories = [
            m for m in memories
            if m.get("created_at", "") < cutoff
            and m.get("importance", 5) < 7
        ]

        # Activities consolidate faster (7 days instead of 30)
        activity_cutoff = (
            datetime.now(timezone.utc) - timedelta(days=7)
        ).isoformat()
        old_activities = [
            m for m in memories
            if m.get("category") == "activity"
            and m.get("created_at", "") < activity_cutoff
            and m not in old_memories
        ]
        old_memories.extend(old_activities)

        if len(old_memories) < 5:
            return 0

        # Group by category and look for duplicates
        consolidated = 0
        seen_content: dict[str, str] = {}

        for m in old_memories:
            content = m.get("content", "")
            category = m.get("category", "")
            key = f"{category}:{content[:50].lower()}"

            if key in seen_content:
                # Mark as superseded
                try:
                    from agents.seo_agent.tools.supabase_tools import upsert_record

                    upsert_record(
                        "ralf_memory",
                        {
                            "id": m["id"],
                            "superseded_by": seen_content[key],
                        },
                        on_conflict="id",
                    )
                    consolidated += 1
                except Exception:
                    pass
            else:
                seen_content[key] = m.get("id", "")

        if consolidated:
            logger.info("Consolidated %d old memories", consolidated)
            self._cache = None

        return consolidated

    def _get_all_active(self) -> list[dict[str, Any]]:
        """Get all non-superseded memories (cached)."""
        if self._cache is not None:
            return self._cache

        from agents.seo_agent.tools.supabase_tools import query_table

        try:
            rows = query_table(
                "ralf_memory",
                limit=200,
                order_by="importance",
                order_desc=True,
            )
            self._cache = [r for r in rows if not r.get("superseded_by")]
            return self._cache
        except Exception:
            return []

    def _increment_recall(self, memory_id: str) -> None:
        """Bump the recall count for a memory."""
        if not memory_id:
            return
        try:
            from agents.seo_agent.tools.supabase_tools import query_table, upsert_record

            rows = query_table("ralf_memory", filters={"id": memory_id}, limit=1)
            if rows:
                current = rows[0].get("recall_count", 0)
                upsert_record(
                    "ralf_memory",
                    {"id": memory_id, "recall_count": current + 1},
                    on_conflict="id",
                )
        except Exception:
            pass


def _similar_content(a: str, b: str) -> bool:
    """Check if two memory content strings are similar enough to be duplicates."""
    a_words = set(a.lower().split())
    b_words = set(b.lower().split())
    if not a_words or not b_words:
        return False
    overlap = len(a_words & b_words)
    return overlap / max(len(a_words), len(b_words)) > 0.6
