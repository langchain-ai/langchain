"""SEO strategy engine — gives Ralf goals, priorities, and initiative.

This module defines the overall SEO strategy, tracks progress against goals,
and generates prioritised next-step recommendations. It also manages the
activity schedule that Ralf follows on every heartbeat.

The schedule is stored in Supabase (``ralf_schedule`` table) so it can be
edited via Telegram.  On first boot the table is seeded from
``DEFAULT_SCHEDULE``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strategic goals — what we're trying to achieve
# ---------------------------------------------------------------------------

GOALS: list[dict[str, Any]] = [
    {
        "id": "organic_traffic",
        "description": "Grow organic traffic across all 3 sites",
        "metric": "monthly organic sessions",
        "targets": {
            "3_month": "1,000 sessions/month combined",
            "6_month": "5,000 sessions/month combined",
            "12_month": "20,000 sessions/month combined",
        },
    },
    {
        "id": "domain_authority",
        "description": "Build domain authority through quality backlinks",
        "metric": "Ahrefs Domain Rating",
        "targets": {
            "3_month": "DR 10+ for all sites",
            "6_month": "DR 20+ for freeroomplanner, DR 15+ for others",
            "12_month": "DR 30+ for freeroomplanner",
        },
    },
    {
        "id": "keyword_rankings",
        "description": "Rank for high-intent keywords",
        "metric": "keywords in top 10",
        "targets": {
            "3_month": "5 keywords in top 10",
            "6_month": "20 keywords in top 10",
            "12_month": "50 keywords in top 10",
        },
    },
    {
        "id": "content_library",
        "description": "Build comprehensive content library",
        "metric": "published blog posts + guides",
        "targets": {
            "3_month": "15 posts across all sites",
            "6_month": "40 posts",
            "12_month": "100 posts",
        },
    },
    {
        "id": "backlink_pipeline",
        "description": "Establish repeatable outreach pipeline",
        "metric": "backlinks acquired per month",
        "targets": {
            "3_month": "5 backlinks/month",
            "6_month": "15 backlinks/month",
            "12_month": "30 backlinks/month",
        },
    },
]

# ---------------------------------------------------------------------------
# Activity schedule — the recurring cadences Ralf follows
# ---------------------------------------------------------------------------
#
# The canonical schedule lives in the ``ralf_schedule`` Supabase table so it
# can be edited via Telegram.  ``DEFAULT_SCHEDULE`` below is the seed data
# that populates the table on first boot.
#
# Cadences:
#   daily   — boosted every matching day-of-week (0=Mon … 6=Sun)
#   weekly  — boosted once per week on a specific day
#   monthly — boosted once per month around a specific day (±1 day)

_DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

DEFAULT_SCHEDULE: list[dict[str, Any]] = [
    # --- Daily focus rotation ---
    {
        "cadence": "daily",
        "day_of_week": 0,
        "skill": "keyword_research",
        "boost_amount": 30,
        "label": "Keyword Research Day",
        "description": "Discover keyword opportunities and analyse content gaps.",
    },
    {
        "cadence": "daily",
        "day_of_week": 0,
        "skill": "keyword_refresh",
        "boost_amount": 30,
        "label": "Keyword Research Day",
        "description": "Discover keyword opportunities and analyse content gaps.",
    },
    {
        "cadence": "daily",
        "day_of_week": 0,
        "skill": "content_gap_analysis",
        "boost_amount": 30,
        "label": "Keyword Research Day",
        "description": "Discover keyword opportunities and analyse content gaps.",
    },
    {
        "cadence": "daily",
        "day_of_week": 1,
        "skill": "publish_blog",
        "boost_amount": 30,
        "label": "Content Writing Day",
        "description": "Write and publish blog posts from top keywords.",
    },
    {
        "cadence": "daily",
        "day_of_week": 2,
        "skill": "publish_blog",
        "boost_amount": 30,
        "label": "Content Writing Day",
        "description": "Continue publishing blog content.",
    },
    {
        "cadence": "daily",
        "day_of_week": 3,
        "skill": "discover_prospects",
        "boost_amount": 30,
        "label": "Backlink Prospecting Day",
        "description": "Find, score, and promote backlink prospects.",
    },
    {
        "cadence": "daily",
        "day_of_week": 3,
        "skill": "score_prospects",
        "boost_amount": 30,
        "label": "Backlink Prospecting Day",
        "description": "Find, score, and promote backlink prospects.",
    },
    {
        "cadence": "daily",
        "day_of_week": 3,
        "skill": "promote_to_crm",
        "boost_amount": 30,
        "label": "Backlink Prospecting Day",
        "description": "Find, score, and promote backlink prospects.",
    },
    {
        "cadence": "daily",
        "day_of_week": 4,
        "skill": "track_rankings",
        "boost_amount": 30,
        "label": "Reporting & Analytics Day",
        "description": "Track rankings, write journal, review weekly progress.",
    },
    {
        "cadence": "daily",
        "day_of_week": 4,
        "skill": "journal_entry",
        "boost_amount": 30,
        "label": "Reporting & Analytics Day",
        "description": "Track rankings, write journal, review weekly progress.",
    },
    {
        "cadence": "daily",
        "day_of_week": 5,
        "skill": "internal_linking",
        "boost_amount": 25,
        "label": "Maintenance Day",
        "description": "Internal linking audit and memory cleanup.",
    },
    {
        "cadence": "daily",
        "day_of_week": 5,
        "skill": "memory_consolidation",
        "boost_amount": 25,
        "label": "Maintenance Day",
        "description": "Internal linking audit and memory cleanup.",
    },
    {
        "cadence": "daily",
        "day_of_week": 5,
        "skill": "memory_promotion",
        "boost_amount": 25,
        "label": "Maintenance Day",
        "description": "Internal linking audit and memory cleanup.",
    },
    {
        "cadence": "daily",
        "day_of_week": 6,
        "skill": "internal_linking",
        "boost_amount": 25,
        "label": "Maintenance Day",
        "description": "Light maintenance and memory promotion.",
    },
    {
        "cadence": "daily",
        "day_of_week": 6,
        "skill": "memory_consolidation",
        "boost_amount": 25,
        "label": "Maintenance Day",
        "description": "Light maintenance and memory promotion.",
    },
    {
        "cadence": "daily",
        "day_of_week": 6,
        "skill": "memory_promotion",
        "boost_amount": 25,
        "label": "Maintenance Day",
        "description": "Light maintenance and memory promotion.",
    },
    # --- Weekly tasks ---
    {
        "cadence": "weekly",
        "day_of_week": 5,
        "skill": "internal_linking",
        "boost_amount": 40,
        "label": "Weekly Link Audit",
        "description": "Audit internal links across all sites.",
    },
    {
        "cadence": "weekly",
        "day_of_week": 6,
        "skill": "memory_consolidation",
        "boost_amount": 40,
        "label": "Weekly Memory Cleanup",
        "description": "Merge old low-importance memories.",
    },
    {
        "cadence": "weekly",
        "day_of_week": 6,
        "skill": "memory_promotion",
        "boost_amount": 40,
        "label": "Weekly Memory Promotion",
        "description": "Promote high-value learnings to permanent lessons.",
    },
    # --- Monthly tasks ---
    {
        "cadence": "monthly",
        "day_of_month": 1,
        "skill": "keyword_refresh",
        "boost_amount": 40,
        "label": "Monthly Keyword Refresh",
        "description": "Full keyword refresh across all sites on the 1st.",
    },
    {
        "cadence": "monthly",
        "day_of_month": 15,
        "skill": "content_gap_analysis",
        "boost_amount": 40,
        "label": "Mid-month Gap Analysis",
        "description": "Mid-month content gap analysis vs competitors.",
    },
]


# ---------------------------------------------------------------------------
# Schedule helpers — read/write the DB-backed schedule
# ---------------------------------------------------------------------------


def seed_schedule() -> int:
    """Populate ``ralf_schedule`` from ``DEFAULT_SCHEDULE`` if the table is empty.

    Returns:
        Number of rows inserted (0 if already seeded).
    """
    from agents.seo_agent.tools.supabase_tools import insert_record, query_table

    existing = query_table("ralf_schedule", limit=1)
    if existing:
        return 0

    count = 0
    for entry in DEFAULT_SCHEDULE:
        row = {**entry}
        row.setdefault("active", True)
        insert_record("ralf_schedule", row)
        count += 1
    logger.info("Seeded ralf_schedule with %d entries", count)
    return count


def get_todays_schedule(now: datetime | None = None) -> dict[str, Any]:
    """Return the combined schedule for today (daily + weekly + monthly boosts).

    Reads from Supabase ``ralf_schedule`` table, falling back to
    ``DEFAULT_SCHEDULE`` if the table is empty.  Merges daily focus boosts
    with any weekly/monthly tasks due today into a single dict with
    ``boost_skills`` mapping skill names to their total boost.

    Args:
        now: Override for current time (useful for testing).

    Returns:
        Dict with ``label``, ``description``, ``boost_skills`` (dict of
        skill name to boost amount), and ``weekly_due`` / ``monthly_due`` lists.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    rows = _load_schedule_rows()

    boosts: dict[str, int] = {}
    label = ""
    description = ""
    weekly_due: list[str] = []
    monthly_due: list[str] = []

    for row in rows:
        if not row.get("active", True):
            continue

        cadence = row.get("cadence", "")
        skill = row.get("skill", "")
        boost = row.get("boost_amount", 30)

        if cadence == "daily" and row.get("day_of_week") == now.weekday():
            boosts[skill] = max(boosts.get(skill, 0), boost)
            if not label:
                label = row.get("label", "")
                description = row.get("description", "")

        elif cadence == "weekly" and row.get("day_of_week") == now.weekday():
            boosts[skill] = max(boosts.get(skill, 0), boost)
            weekly_due.append(skill)

        elif cadence == "monthly":
            target_day = row.get("day_of_month", 1)
            if abs(now.day - target_day) <= 1:
                boosts[skill] = max(boosts.get(skill, 0), boost)
                monthly_due.append(skill)

    if not label:
        label = f"{_DAY_NAMES[now.weekday()]} (no focus set)"
        description = "No specific focus scheduled for today."

    return {
        "label": label,
        "description": description,
        "boost_skills": boosts,
        "weekly_due": weekly_due,
        "monthly_due": monthly_due,
    }


def _load_schedule_rows() -> list[dict[str, Any]]:
    """Load schedule rows from Supabase, seeding defaults if empty.

    Returns:
        List of active schedule entry dicts.
    """
    from agents.seo_agent.tools.supabase_tools import query_table

    rows = query_table("ralf_schedule", limit=200)
    if not rows:
        seed_schedule()
        rows = query_table("ralf_schedule", limit=200)
    # Filter active entries (handles both bool and string representations)
    return [r for r in rows if r.get("active", True) is not False]


def get_full_schedule() -> list[dict[str, Any]]:
    """Return all active schedule entries, grouped by cadence and day.

    Returns:
        List of schedule entry dicts sorted by cadence then day.
    """
    rows = _load_schedule_rows()
    rows.sort(key=lambda r: (
        {"daily": 0, "weekly": 1, "monthly": 2}.get(r.get("cadence", ""), 3),
        r.get("day_of_week", 0),
        r.get("day_of_month", 0),
    ))
    return rows


def update_schedule_entry(entry_id: str, **updates: Any) -> dict[str, Any]:
    """Update a schedule row in the database.

    Args:
        entry_id: UUID of the schedule entry.
        **updates: Fields to update (e.g. ``day_of_week=2``, ``active=False``).

    Returns:
        The updated record dict.
    """
    from agents.seo_agent.tools.supabase_tools import upsert_record

    updates["updated_at"] = datetime.now(timezone.utc).isoformat()
    return upsert_record("ralf_schedule", {"id": entry_id, **updates})


def log_schedule_completion(
    skill: str,
    *,
    site: str = "",
    summary: str = "",
    heartbeat_id: str = "",
    status: str = "done",
    schedule_date: str | None = None,
) -> dict[str, Any]:
    """Record a completed (or failed/skipped) schedule activity.

    Args:
        skill: Skill name that was executed.
        site: Target site key.
        summary: Short description of what was done.
        heartbeat_id: WAL cycle ID for correlation.
        status: Completion status (``done``, ``failed``, ``skipped``).
        schedule_date: ISO date string; defaults to today UTC.

    Returns:
        The inserted log record.
    """
    from agents.seo_agent.tools.supabase_tools import insert_record

    if not schedule_date:
        schedule_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    data: dict[str, Any] = {
        "schedule_date": schedule_date,
        "skill": skill,
        "status": status,
        "site": site,
        "summary": summary,
        "heartbeat_id": heartbeat_id,
    }
    if status == "done":
        data["completed_at"] = datetime.now(timezone.utc).isoformat()

    return insert_record("ralf_schedule_log", data)


def get_schedule_history(days_back: int = 7) -> list[dict[str, Any]]:
    """Query recent schedule completion history.

    Args:
        days_back: How many days to look back (default 7).

    Returns:
        List of schedule log entries, newest first.
    """
    from datetime import timedelta

    from agents.seo_agent.tools.supabase_tools import query_table

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")

    # query_table only supports equality filters, so we fetch more and filter
    rows = query_table(
        "ralf_schedule_log",
        limit=500,
        order_by="created_at",
        order_desc=True,
    )
    return [r for r in rows if r.get("schedule_date", "") >= cutoff]


def get_todays_log() -> list[dict[str, Any]]:
    """Return today's schedule log entries.

    Returns:
        List of log entries for today, newest first.
    """
    from agents.seo_agent.tools.supabase_tools import query_table

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return query_table(
        "ralf_schedule_log",
        filters={"schedule_date": today},
        limit=50,
        order_by="created_at",
        order_desc=True,
    )


def format_schedule_for_display(rows: list[dict[str, Any]]) -> str:
    """Format schedule entries as a human-readable string for Telegram.

    Args:
        rows: Schedule entries from ``get_full_schedule()``.

    Returns:
        Formatted schedule string.
    """
    lines: list[str] = []
    current_cadence = ""

    for row in rows:
        cadence = row.get("cadence", "")
        if cadence != current_cadence:
            current_cadence = cadence
            lines.append(f"\n{'Daily' if cadence == 'daily' else cadence.title()} Schedule:")
            lines.append("-" * 30)

        skill = row.get("skill", "")
        boost = row.get("boost_amount", 0)
        active = row.get("active", True)
        status = "" if active else " [PAUSED]"

        if cadence == "daily":
            day = _DAY_NAMES[row.get("day_of_week", 0)]
            lines.append(f"  {day}: {skill} (+{boost}){status}")
        elif cadence == "weekly":
            day = _DAY_NAMES[row.get("day_of_week", 0)]
            lines.append(f"  Every {day}: {skill} (+{boost}){status}")
        elif cadence == "monthly":
            dom = row.get("day_of_month", 1)
            suffix = "th" if 4 <= dom <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(dom % 10, "th")
            lines.append(f"  {dom}{suffix} of month: {skill} (+{boost}){status}")

    return "\n".join(lines) if lines else "No schedule entries found."

# ---------------------------------------------------------------------------
# Priority scoring — what to work on next
# ---------------------------------------------------------------------------

# Content priority: keywords sorted by opportunity score
# Score = volume / (difficulty + 1) * intent_multiplier
CONTENT_RULES: dict[str, Any] = {
    "max_same_topic_in_row": 1,   # Never publish 2+ posts on the same topic cluster consecutively
    "topic_cooldown_days": 7,     # Wait at least 7 days before revisiting a topic cluster
    "category_targets": {         # Aim for this distribution
        "room_planning": 0.30,    # 30% room planning / floor plans
        "kitchen": 0.25,          # 25% kitchen content
        "bathroom": 0.15,         # 15% bathroom content
        "bedroom": 0.10,          # 10% bedroom content
        "extensions": 0.10,       # 10% extensions / renovations
        "general": 0.10,          # 10% general home improvement
    },
}

# Content priority: keywords sorted by opportunity score
# Score = volume / (difficulty + 1) * intent_multiplier
INTENT_MULTIPLIERS: dict[str, float] = {
    "transactional": 3.0,
    "commercial": 2.5,
    "informational": 1.0,
    "navigational": 0.3,
    "unknown": 1.0,
}


def score_keyword_opportunity(volume: int, difficulty: int, intent: str = "unknown") -> float:
    """Score a keyword by its opportunity value.

    Higher volume, lower difficulty, and commercial intent = higher score.
    """
    multiplier = INTENT_MULTIPLIERS.get(intent.lower(), 1.0)
    return (volume / (difficulty + 1)) * multiplier


def get_site_priority() -> list[dict[str, str]]:
    """Return sites in priority order based on current state.

    freeroomplanner has the most content and highest potential — prioritise it.
    kitchensdirectory has the most listings but needs content.
    kitchen_estimator is newest and needs everything.
    """
    return [
        {
            "site": "freeroomplanner",
            "priority": "HIGH",
            "reason": "Most content, highest traffic potential. Room planner keywords have strong volume.",
            "focus": "Publish 2 blog posts/week. Target 'free room planner', 'kitchen layout planner', 'floor plan maker'.",
        },
        {
            "site": "kitchensdirectory",
            "priority": "MEDIUM",
            "reason": "159 makers listed, strong domain but zero DR. Needs content + backlinks.",
            "focus": "Location pages (kitchen makers in [city]). Backlink outreach to home improvement sites.",
        },
        {
            "site": "kitchen_estimator",
            "priority": "MEDIUM",
            "reason": "High-intent tool but new domain. Cost calculator keywords are competitive.",
            "focus": "Regional cost guides (kitchen cost in [region]). Capture calculator traffic.",
        },
    ]


def generate_next_steps(
    existing_keywords: int,
    existing_content: int,
    existing_prospects: int,
    existing_gaps: int,
) -> list[str]:
    """Generate prioritised next steps based on current database state."""
    steps: list[str] = []

    if existing_keywords == 0:
        steps.append("Run keyword research for all sites — we need data to work with.")
    elif existing_content == 0:
        steps.append(f"We have {existing_keywords} keywords but no content. Create content briefs for the top 5 keywords by opportunity score.")
    elif existing_content < 10:
        steps.append(f"We have {existing_content} content pieces. Write 2-3 more blog posts targeting our highest-opportunity keywords.")
    elif existing_content < 30:
        steps.append(f"{existing_content} posts published. Keep up the pace — target 2 posts/week. Focus on freeroomplanner (highest potential).")

    if existing_gaps == 0 and existing_keywords > 0:
        steps.append("Run content gap analysis to find what competitors rank for that we don't.")

    if existing_prospects == 0:
        steps.append(
            "Start backlink prospecting. Priority: kitchen/bathroom providers for room planner embeds "
            "(partnership approach), then home interior bloggers (content collaboration). "
            "These give us the most natural, high-value links."
        )
    elif existing_prospects > 0 and existing_prospects < 20:
        steps.append(
            f"We have {existing_prospects} prospects. Score them, classify by segment "
            f"(provider/blogger/influencer/resource/PR), then generate tailored outreach emails."
        )
    elif existing_prospects >= 20:
        steps.append(
            f"{existing_prospects} prospects in pipeline. Check who needs follow-up, "
            f"generate emails for uncontacted tier-1 prospects, and review any replies."
        )

    # Always include a strategic reminder
    if existing_content > 0 and existing_prospects > 0:
        steps.append(
            "Track rankings for all sites to measure progress. Compare against "
            "3-month targets: 5 keywords in top 10, DR 10+ for all sites."
        )

    if not steps:
        steps.append("Review weekly report and compare progress against our 3-month targets.")

    return steps


def get_strategy_summary() -> str:
    """Generate a strategy summary Ralf can include in conversations."""
    priorities = get_site_priority()
    summary = "Current SEO Strategy:\n\n"
    for p in priorities:
        summary += f"[{p['priority']}] {p['site']}: {p['focus']}\n"
    summary += "\nGoals: Grow to 5,000 organic sessions/month in 6 months. DR 20+ for freeroomplanner."
    return summary
