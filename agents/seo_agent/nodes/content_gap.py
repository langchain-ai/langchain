"""Content gap analysis node — finds keywords competitors rank for but we do not.

Runs an Ahrefs content gap analysis against the top competing domains and
categorises each gap keyword by funnel stage.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from agents.seo_agent.config import SITE_PROFILES
from agents.seo_agent.state import SEOAgentState
from agents.seo_agent.tools import ahrefs_tools, supabase_tools

logger = logging.getLogger(__name__)

# Maximum number of competitors to include in the gap analysis
_MAX_COMPETITORS = 5

# Funnel-stage classification patterns
_AWARENESS_PATTERNS = re.compile(
    r"\b(how[ -]to|what[ -]is|guide|tips|ideas|ways to|explained|tutorial)\b",
    re.IGNORECASE,
)
_CONSIDERATION_PATTERNS = re.compile(
    r"\b(best|compare|comparison|vs|versus|review|cost|price|top \d|alternatives)\b",
    re.IGNORECASE,
)
_DECISION_PATTERNS = re.compile(
    r"\b(near me|quote|hire|buy|book|get a|install|fitted|local)\b",
    re.IGNORECASE,
)


def _classify_funnel_stage(keyword: str) -> str:
    """Classify a keyword into a marketing funnel stage.

    Uses pattern matching to determine whether the keyword signals
    awareness, consideration, or decision intent.

    Args:
        keyword: The keyword string to classify.

    Returns:
        One of ``"awareness"``, ``"consideration"``, or ``"decision"``.
    """
    if _DECISION_PATTERNS.search(keyword):
        return "decision"
    if _CONSIDERATION_PATTERNS.search(keyword):
        return "consideration"
    if _AWARENESS_PATTERNS.search(keyword):
        return "awareness"
    # Default to awareness for informational keywords without clear signals
    return "awareness"


def run_content_gap(state: SEOAgentState) -> dict[str, Any]:
    """Identify content gap keywords between the target site and competitors.

    Discovers competing domains via Ahrefs, runs a content gap analysis
    against the top 5, categorises each keyword by funnel stage, and
    persists the results to Supabase.

    Args:
        state: The current SEO agent state.

    Returns:
        State update with `content_gaps`, `errors`, and `next_node`.
    """
    errors: list[str] = list(state.get("errors", []))
    content_gaps: list[dict[str, Any]] = []
    target_site = state["target_site"]

    profile = SITE_PROFILES.get(target_site)
    if profile is None:
        msg = f"No site profile found for '{target_site}'"
        logger.error(msg)
        errors.append(msg)
        return {
            "content_gaps": [],
            "errors": errors,
            "next_node": "END",
        }

    domain = profile.get("domain", "")
    if not domain:
        msg = f"No domain configured for '{target_site}'"
        logger.error(msg)
        errors.append(msg)
        return {
            "content_gaps": [],
            "errors": errors,
            "next_node": "END",
        }

    # Step 1: Discover competing domains from Ahrefs
    try:
        competing_domains_raw = ahrefs_tools.get_competing_domains.invoke(domain)
    except Exception:
        msg = f"Failed to fetch competing domains for '{domain}'"
        logger.warning(msg, exc_info=True)
        errors.append(msg)
        # Fall back to competitors listed in the site profile
        competing_domains_raw = [
            {"domain": d} for d in profile.get("competitors", [])
        ]

    competitor_domains = [
        entry["domain"]
        for entry in competing_domains_raw[:_MAX_COMPETITORS]
        if entry.get("domain")
    ]

    if not competitor_domains:
        msg = f"No competitor domains found for '{target_site}'"
        logger.warning(msg)
        errors.append(msg)
        return {
            "content_gaps": [],
            "errors": errors,
            "next_node": "END",
        }

    logger.info(
        "Running content gap: %s vs %s",
        domain,
        competitor_domains,
    )

    # Step 2: Run the content gap analysis
    try:
        gap_keywords = ahrefs_tools.get_content_gap.invoke(
            {"target": domain, "competitors": competitor_domains}
        )
    except Exception:
        msg = f"Ahrefs content gap analysis failed for '{domain}'"
        logger.error(msg, exc_info=True)
        errors.append(msg)
        return {
            "content_gaps": [],
            "errors": errors,
            "next_node": "END",
        }

    # Step 3: Categorise each gap keyword and persist
    for gap in gap_keywords:
        keyword_text = gap.get("keyword", "")
        if not keyword_text:
            continue

        funnel_stage = _classify_funnel_stage(keyword_text)

        gap_record: dict[str, Any] = {
            "keyword": keyword_text,
            "volume": gap.get("volume", 0),
            "kd": gap.get("kd", 0),
            "funnel_stage": funnel_stage,
            "competitors_ranking": gap.get("competitors_ranking", []),
            "top_url": gap.get("top_url", ""),
            "target_site": target_site,
            "competitor_source": ", ".join(competitor_domains),
        }
        content_gaps.append(gap_record)

        # Save to Supabase
        try:
            supabase_tools.insert_record("seo_content_gaps", gap_record)
        except Exception:
            msg = f"Failed to save content gap '{keyword_text}' to Supabase"
            logger.warning(msg, exc_info=True)
            errors.append(msg)

    # Log summary by funnel stage
    stage_counts: dict[str, int] = {}
    for gap in content_gaps:
        stage = gap.get("funnel_stage", "unknown")
        stage_counts[stage] = stage_counts.get(stage, 0) + 1

    logger.info(
        "Content gap analysis complete for %s: %d gaps (%s)",
        target_site,
        len(content_gaps),
        ", ".join(f"{k}: {v}" for k, v in stage_counts.items()),
    )

    return {
        "content_gaps": content_gaps,
        "errors": errors,
        "next_node": "END",
    }
