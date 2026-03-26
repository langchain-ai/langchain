"""SEO strategy engine — gives Ralf goals, priorities, and initiative.

This module defines the overall SEO strategy, tracks progress against goals,
and generates prioritised next-step recommendations. Ralf consults this
before every conversation to know what matters most right now.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

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
# SEO playbook — the recurring process Ralf follows
# ---------------------------------------------------------------------------

WEEKLY_PLAYBOOK: list[dict[str, str]] = [
    {
        "day": "Monday",
        "task": "keyword_research",
        "description": "Run keyword research for all sites. Identify new opportunities. Compare against existing content to find gaps.",
        "output": "New keyword opportunities saved to database",
    },
    {
        "day": "Tuesday",
        "task": "content_planning",
        "description": "Review keyword opportunities. Generate content briefs for top 3-5 unaddressed keywords. Prioritise by volume/difficulty ratio.",
        "output": "Content briefs ready for writing",
    },
    {
        "day": "Wednesday",
        "task": "content_writing",
        "description": "Write and publish 1-2 blog posts from the content briefs. Target the highest-opportunity keywords first.",
        "output": "Blog posts published to sites",
    },
    {
        "day": "Thursday",
        "task": "backlink_prospecting",
        "description": "Discover new backlink prospects. Enrich and score existing prospects. Generate outreach emails for tier 1 prospects.",
        "output": "Outreach emails ready for review",
    },
    {
        "day": "Friday",
        "task": "reporting",
        "description": "Generate weekly report: rankings, traffic, content published, backlinks acquired, spend. Compare to goals.",
        "output": "Weekly report sent",
    },
]

# ---------------------------------------------------------------------------
# Priority scoring — what to work on next
# ---------------------------------------------------------------------------

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
