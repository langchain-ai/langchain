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
    """Classify a keyword into a marketing funnel stage."""
    if _DECISION_PATTERNS.search(keyword):
        return "decision"
    if _CONSIDERATION_PATTERNS.search(keyword):
        return "consideration"
    if _AWARENESS_PATTERNS.search(keyword):
        return "awareness"
    return "awareness"


def _run_for_single_site(
    target_site: str, profile: dict, errors: list[str]
) -> list[dict[str, Any]]:
    """Run content gap analysis for a single site."""
    content_gaps: list[dict[str, Any]] = []
    domain = profile.get("domain", "")
    if not domain:
        errors.append(f"No domain configured for '{target_site}'")
        return content_gaps

    # Step 1: Discover competing domains from Ahrefs
    try:
        competing_domains_raw = ahrefs_tools.get_competing_domains.invoke(domain)
    except Exception:
        msg = f"Failed to fetch competing domains for '{domain}'"
        logger.warning(msg, exc_info=True)
        errors.append(msg)
        competing_domains_raw = [
            {"domain": d} for d in profile.get("competitors", [])
        ]

    competitor_domains = [
        entry.get("competitor_domain") or entry.get("domain", "")
        for entry in competing_domains_raw[:_MAX_COMPETITORS]
        if entry.get("competitor_domain") or entry.get("domain")
    ]

    if not competitor_domains:
        errors.append(f"No competitor domains found for '{target_site}'")
        return content_gaps

    logger.info("Running content gap: %s vs %s", domain, competitor_domains)

    # Step 2: Run the content gap analysis
    try:
        gap_keywords = ahrefs_tools.get_content_gap.invoke(
            {"target": domain, "competitors": competitor_domains}
        )
    except Exception:
        msg = f"Ahrefs content gap analysis failed for '{domain}'"
        logger.error(msg, exc_info=True)
        errors.append(msg)
        return content_gaps

    # Step 3: Categorise each gap keyword and persist
    for gap in gap_keywords:
        keyword_text = gap.get("keyword", "")
        if not keyword_text:
            continue

        funnel_stage = _classify_funnel_stage(keyword_text)

        gap_record: dict[str, Any] = {
            "keyword": keyword_text,
            "volume": gap.get("volume", 0),
            "kd": gap.get("difficulty") or gap.get("keyword_difficulty") or gap.get("kd") or 0,
            "funnel_stage": funnel_stage,
            "competitors_ranking": gap.get("competitors_ranking", []),
            "top_url": gap.get("top_url", ""),
            "target_site": target_site,
            "competitor_source": ", ".join(competitor_domains),
        }
        content_gaps.append(gap_record)

        try:
            supabase_tools.insert_record("seo_content_gaps", gap_record)
        except Exception:
            msg = f"Failed to save content gap '{keyword_text}' to Supabase"
            logger.warning(msg, exc_info=True)
            errors.append(msg)

    logger.info(
        "Content gap for %s: %d gaps found",
        target_site,
        len(content_gaps),
    )
    return content_gaps


def run_content_gap(state: SEOAgentState) -> dict[str, Any]:
    """Identify content gap keywords between the target site and competitors."""
    errors: list[str] = list(state.get("errors", []))
    content_gaps: list[dict[str, Any]] = []
    target_site = state["target_site"]

    # Handle "all" — run for every site profile
    if target_site == "all":
        sites_to_run = list(SITE_PROFILES.keys())
    else:
        sites_to_run = [target_site]

    for site_key in sites_to_run:
        profile = SITE_PROFILES.get(site_key)
        if profile is None:
            errors.append(f"No site profile found for '{site_key}'")
            continue
        gaps = _run_for_single_site(site_key, profile, errors)
        content_gaps.extend(gaps)

    stage_counts: dict[str, int] = {}
    for gap in content_gaps:
        stage = gap.get("funnel_stage", "unknown")
        stage_counts[stage] = stage_counts.get(stage, 0) + 1

    logger.info(
        "Content gap complete: %d gaps across %s (%s)",
        len(content_gaps),
        sites_to_run,
        ", ".join(f"{k}: {v}" for k, v in stage_counts.items()),
    )

    return {
        "content_gaps": content_gaps,
        "errors": errors,
        "next_node": "END",
    }
