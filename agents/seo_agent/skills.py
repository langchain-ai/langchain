"""Self-describing skill registry for autonomous invocation.

Each skill describes when it should fire, what preconditions it needs,
its priority, cost tier, cooldown, and what it produces. The heartbeat
uses this registry to dynamically decide what to do next, rather than
relying on a hardcoded decision tree.

Skills can be triggered by:
    - ``preconditions``: Data-driven triggers (e.g. "keywords_discovered == 0")
    - ``schedule``: Time-based triggers (e.g. "every 3 days", "fridays")
    - ``reactive``: Event-driven triggers (e.g. "new_prospects_unscored > 5")

Usage::

    from agents.seo_agent.skills import SkillRegistry

    registry = SkillRegistry()
    dashboard = get_dashboard_summary()
    buffer = WorkingBuffer()

    # Get prioritised list of skills that should fire right now
    actionable = registry.evaluate(dashboard, buffer)

    for skill in actionable:
        print(f"{skill.name} (priority={skill.priority}, cost={skill.cost_tier})")
        # Execute via skill.execute(...)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    """A self-describing agent skill with metadata for autonomous invocation.

    Attributes:
        name: Unique skill identifier (matches task_type where applicable).
        description: Human-readable description of what this skill does.
        category: Grouping (content, prospecting, analytics, maintenance).
        priority: Base priority 1-100 (higher = more important).
        cost_tier: LLM cost tier (haiku, sonnet, opus, none).
        cooldown_hours: Minimum hours between invocations.
        preconditions: List of callables that check dashboard/buffer state.
            Each returns (should_fire: bool, reason: str).
        produces: What this skill outputs (for dependency tracking).
        consumes: What this skill needs as input.
        task_type: The LangGraph task_type to invoke, if graph-based.
        execute_fn: Optional custom execution function (for non-graph skills).
        sites: Which sites this skill applies to ("all" or list of site keys).
        autonomous: Whether this skill can fire without user confirmation.
    """

    name: str
    description: str
    category: str
    priority: int
    cost_tier: str = "sonnet"
    cooldown_hours: int = 6
    preconditions: list[Callable] = field(default_factory=list)
    produces: list[str] = field(default_factory=list)
    consumes: list[str] = field(default_factory=list)
    task_type: str = ""
    execute_fn: Callable | None = None
    sites: str | list[str] = "all"
    autonomous: bool = True

    def should_fire(
        self,
        dashboard: dict[str, Any],
        buffer: Any,
    ) -> tuple[bool, str]:
        """Evaluate whether this skill should fire now.

        Args:
            dashboard: Current dashboard summary from ``get_dashboard_summary()``.
            buffer: WorkingBuffer instance for ephemeral state.

        Returns:
            Tuple of (should_fire, reason).
        """
        # Check cooldown
        last_run_key = f"skill_last_run_{self.name}"
        last_run = buffer.get(last_run_key)
        if last_run:
            from datetime import timedelta

            now = datetime.now(timezone.utc)
            try:
                last_dt = datetime.fromisoformat(last_run)
                if (now - last_dt) < timedelta(hours=self.cooldown_hours):
                    hours_left = self.cooldown_hours - (now - last_dt).total_seconds() / 3600
                    return False, f"Cooldown: {hours_left:.1f}h remaining"
            except (ValueError, TypeError):
                pass

        # Check all preconditions
        for check in self.preconditions:
            try:
                should, reason = check(dashboard, buffer)
                if not should:
                    return False, reason
            except Exception as e:
                return False, f"Precondition error: {e}"

        return True, "All preconditions met"

    def record_execution(self, buffer: Any) -> None:
        """Record that this skill was executed (for cooldown tracking).

        Args:
            buffer: WorkingBuffer instance.
        """
        buffer.set(
            f"skill_last_run_{self.name}",
            datetime.now(timezone.utc).isoformat(),
            ttl_hours=max(self.cooldown_hours * 2, 48),
        )


# ---------------------------------------------------------------------------
# Precondition builders (reusable checks)
# ---------------------------------------------------------------------------


def needs_keywords(dashboard: dict, buffer: Any) -> tuple[bool, str]:
    """Fire when no keywords have been discovered."""
    count = dashboard.get("keywords_discovered", 0)
    if count == 0:
        return True, "No keywords discovered yet"
    return False, f"{count} keywords already exist"


def has_keywords(dashboard: dict, buffer: Any) -> tuple[bool, str]:
    """Require that some keywords exist."""
    if dashboard.get("keywords_discovered", 0) > 0:
        return True, "Keywords available"
    return False, "No keywords yet"


def needs_content_gaps(dashboard: dict, buffer: Any) -> tuple[bool, str]:
    """Fire when no content gaps have been analysed."""
    if dashboard.get("content_gaps", 0) == 0 and dashboard.get("keywords_discovered", 0) > 0:
        return True, "Keywords exist but no gap analysis done"
    return False, f"{dashboard.get('content_gaps', 0)} gaps already exist"


def needs_prospects(dashboard: dict, buffer: Any) -> tuple[bool, str]:
    """Fire when prospect pipeline is empty."""
    if dashboard.get("prospects_total", 0) == 0:
        return True, "Prospect pipeline is empty"
    return False, f"{dashboard.get('prospects_total', 0)} prospects exist"


def has_unscored_prospects(dashboard: dict, buffer: Any) -> tuple[bool, str]:
    """Fire when there are new/unenriched prospects."""
    pipeline = dashboard.get("prospect_pipeline", {})
    new_count = pipeline.get("new", 0)
    if new_count > 0:
        return True, f"{new_count} unscored prospects"
    return False, "No unscored prospects"


def content_below_target(dashboard: dict, buffer: Any) -> tuple[bool, str]:
    """Fire when content count is below the target threshold."""
    count = dashboard.get("content_pieces", 0)
    target = 30  # configurable via buffer
    override = buffer.get("content_target")
    if override:
        target = int(override)
    if count < target:
        return True, f"{count}/{target} content pieces"
    return False, f"Content target met: {count}/{target}"


def has_content(dashboard: dict, buffer: Any) -> tuple[bool, str]:
    """Require that some content exists."""
    if dashboard.get("content_pieces", 0) > 0:
        return True, "Content exists"
    return False, "No content yet"


def has_prospects(dashboard: dict, buffer: Any) -> tuple[bool, str]:
    """Require that some prospects exist."""
    if dashboard.get("prospects_total", 0) > 0:
        return True, "Prospects exist"
    return False, "No prospects yet"


def journal_due(dashboard: dict, buffer: Any) -> tuple[bool, str]:
    """Fire when a journal entry is due (every ~3 days)."""
    last_journal = buffer.get("last_journal_date")
    if not last_journal:
        return True, "No journal entries recorded in buffer"

    from datetime import timedelta

    try:
        last_dt = datetime.fromisoformat(last_journal)
        days_since = (datetime.now(timezone.utc) - last_dt).days
        if days_since >= 3:
            return True, f"{days_since} days since last journal"
        return False, f"Last journal {days_since} days ago (cooldown: 3 days)"
    except (ValueError, TypeError):
        return True, "Could not parse last journal date"


def not_rate_limited(dashboard: dict, buffer: Any) -> tuple[bool, str]:
    """Check that external APIs are not currently rate-limited."""
    if buffer.get("ahrefs_rate_limited"):
        return False, "Ahrefs is rate-limited"
    if buffer.get("openrouter_rate_limited"):
        return False, "OpenRouter is rate-limited"
    return True, "No rate limits active"


def internal_links_needed(dashboard: dict, buffer: Any) -> tuple[bool, str]:
    """Fire when there are enough posts to benefit from internal linking."""
    count = dashboard.get("content_pieces", 0)
    last_link_audit = buffer.get("last_internal_link_audit")
    if count >= 5 and not last_link_audit:
        return True, f"{count} posts exist, no internal link audit done"
    if last_link_audit:
        from datetime import timedelta

        try:
            last_dt = datetime.fromisoformat(last_link_audit)
            days_since = (datetime.now(timezone.utc) - last_dt).days
            if days_since >= 7 and count >= 5:
                return True, f"{days_since} days since last link audit, {count} posts"
            return False, f"Last link audit {days_since} days ago"
        except (ValueError, TypeError):
            pass
    return False, f"Only {count} posts (need 5+)"


# ---------------------------------------------------------------------------
# Skill registry with all built-in skills
# ---------------------------------------------------------------------------

_BUILTIN_SKILLS: list[Skill] = [
    Skill(
        name="keyword_research",
        description="Discover keyword opportunities via Ahrefs for all active sites",
        category="content",
        priority=90,
        cost_tier="haiku",
        cooldown_hours=24,
        preconditions=[needs_keywords, not_rate_limited],
        produces=["keywords"],
        consumes=[],
        task_type="keyword_research",
        sites="all",
    ),
    Skill(
        name="content_gap_analysis",
        description="Analyse content gaps vs competitors to find unaddressed keywords",
        category="content",
        priority=80,
        cost_tier="sonnet",
        cooldown_hours=48,
        preconditions=[needs_content_gaps, has_keywords, not_rate_limited],
        produces=["content_gaps"],
        consumes=["keywords"],
        task_type="content_gap",
    ),
    Skill(
        name="publish_blog",
        description="Write and publish a blog post targeting the highest-opportunity keyword",
        category="content",
        priority=75,
        cost_tier="sonnet",
        cooldown_hours=8,
        preconditions=[has_keywords, content_below_target],
        produces=["content"],
        consumes=["keywords"],
        task_type="",  # Custom execution, not a direct graph task
    ),
    Skill(
        name="discover_prospects",
        description="Find new backlink prospects via Ahrefs competitor analysis",
        category="prospecting",
        priority=70,
        cost_tier="haiku",
        cooldown_hours=48,
        preconditions=[needs_prospects, has_keywords, not_rate_limited],
        produces=["prospects"],
        consumes=["keywords"],
        task_type="discover_prospects",
    ),
    Skill(
        name="score_prospects",
        description="Enrich and score unprocessed backlink prospects",
        category="prospecting",
        priority=65,
        cost_tier="haiku",
        cooldown_hours=12,
        preconditions=[has_unscored_prospects],
        produces=["scored_prospects"],
        consumes=["prospects"],
        task_type="score_prospects",
    ),
    Skill(
        name="promote_to_crm",
        description="Promote scored prospects to CRM contacts for outreach",
        category="prospecting",
        priority=60,
        cost_tier="none",
        cooldown_hours=6,
        preconditions=[has_prospects],
        produces=["crm_contacts"],
        consumes=["scored_prospects"],
    ),
    Skill(
        name="track_rankings",
        description="Snapshot current search rankings from Ahrefs/GSC",
        category="analytics",
        priority=50,
        cost_tier="none",
        cooldown_hours=24,
        preconditions=[has_content, not_rate_limited],
        produces=["rankings"],
        consumes=["content"],
        task_type="rank_tracker",
    ),
    Skill(
        name="journal_entry",
        description="Write a reflective journal post for ralfseo.com",
        category="content",
        priority=40,
        cost_tier="sonnet",
        cooldown_hours=72,
        preconditions=[journal_due],
        produces=["journal"],
        consumes=[],
        sites=["ralf_seo"],
    ),
    Skill(
        name="internal_linking",
        description="Audit and suggest internal links across blog posts",
        category="maintenance",
        priority=35,
        cost_tier="haiku",
        cooldown_hours=168,  # Weekly
        preconditions=[internal_links_needed],
        produces=["internal_links"],
        consumes=["content"],
        task_type="internal_linker",
    ),
    Skill(
        name="memory_consolidation",
        description="Consolidate old memories to keep the memory store efficient",
        category="maintenance",
        priority=10,
        cost_tier="none",
        cooldown_hours=168,  # Weekly
        preconditions=[],
        produces=[],
        consumes=[],
    ),
    Skill(
        name="keyword_refresh",
        description="Re-run keyword research when all keywords have content",
        category="content",
        priority=70,
        cost_tier="haiku",
        cooldown_hours=48,
        preconditions=[has_keywords, not_rate_limited],
        produces=["keywords"],
        consumes=[],
        task_type="keyword_research",
    ),
]


class SkillRegistry:
    """Registry of all available skills with evaluation and execution support.

    The registry evaluates each skill's preconditions against the current
    dashboard state and working buffer, returning a prioritised list of
    skills that should fire.
    """

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}
        for skill in _BUILTIN_SKILLS:
            self._skills[skill.name] = skill

    def register(self, skill: Skill) -> None:
        """Register a custom skill.

        Args:
            skill: The skill to register.
        """
        self._skills[skill.name] = skill
        logger.info("Registered skill: %s", skill.name)

    def get(self, name: str) -> Skill | None:
        """Get a skill by name.

        Args:
            name: Skill name.

        Returns:
            The skill, or None if not found.
        """
        return self._skills.get(name)

    def all_skills(self) -> list[Skill]:
        """Return all registered skills."""
        return list(self._skills.values())

    def evaluate(
        self,
        dashboard: dict[str, Any],
        buffer: Any,
        *,
        budget_remaining: float = 1.0,
        max_skills: int = 5,
    ) -> list[tuple[Skill, str]]:
        """Evaluate all skills and return those that should fire now.

        Args:
            dashboard: Current dashboard summary.
            buffer: WorkingBuffer instance.
            budget_remaining: Fraction of weekly budget remaining (0.0-1.0).
            max_skills: Maximum skills to return per cycle.

        Returns:
            List of (skill, reason) tuples, sorted by priority descending.
        """
        actionable: list[tuple[Skill, str, int]] = []

        for skill in self._skills.values():
            # Skip non-autonomous skills
            if not skill.autonomous:
                continue

            # Skip expensive skills if budget is low
            if budget_remaining < 0.2 and skill.cost_tier in ("sonnet", "opus"):
                continue
            if budget_remaining < 0.05 and skill.cost_tier != "none":
                continue

            should, reason = skill.should_fire(dashboard, buffer)
            if should:
                # Dynamic priority adjustment
                priority = skill.priority

                # Boost priority for skills that fill gaps in the pipeline
                if skill.category == "content" and dashboard.get("content_pieces", 0) == 0:
                    priority += 20
                if skill.category == "prospecting" and dashboard.get("prospects_total", 0) == 0:
                    priority += 15

                actionable.append((skill, reason, priority))

        # Sort by adjusted priority
        actionable.sort(key=lambda x: x[2], reverse=True)

        return [(s, r) for s, r, _ in actionable[:max_skills]]

    def describe_all(self) -> str:
        """Return a human-readable description of all skills.

        Useful for including in LLM prompts so the agent knows its capabilities.

        Returns:
            Formatted string describing all skills.
        """
        lines = ["Available Skills:"]
        for skill in sorted(self._skills.values(), key=lambda s: -s.priority):
            status = "autonomous" if skill.autonomous else "manual"
            lines.append(
                f"  [{skill.priority:3d}] {skill.name} ({skill.category}, "
                f"{skill.cost_tier}, {status}): {skill.description}"
            )
            if skill.consumes:
                lines.append(f"        needs: {', '.join(skill.consumes)}")
            if skill.produces:
                lines.append(f"        outputs: {', '.join(skill.produces)}")
        return "\n".join(lines)
