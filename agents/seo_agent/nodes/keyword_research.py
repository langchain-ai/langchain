"""Keyword research node — discovers and filters keyword opportunities.

Uses Ahrefs keyword ideas and GSC top queries to find low-competition,
high-volume keywords while avoiding cannibalisation of existing rankings.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from agents.seo_agent.config import SITE_PROFILES
from agents.seo_agent.state import SEOAgentState
from agents.seo_agent.tools import ahrefs_tools, gsc_tools, supabase_tools

logger = logging.getLogger(__name__)

# People Also Ask signal patterns
_PAA_PATTERNS = re.compile(
    r"\b(what|how|why|when|where|which|can|do|does|is|are|should)\b",
    re.IGNORECASE,
)

# Filtering thresholds
_MAX_KD = 40
_MIN_VOLUME = 100
_CANNIBALISATION_MAX_POSITION = 5.0


def _is_paa_keyword(keyword: str) -> bool:
    """Check whether a keyword looks like a People Also Ask question.

    Args:
        keyword: The keyword string to test.

    Returns:
        True if the keyword matches a question-style pattern.
    """
    return bool(_PAA_PATTERNS.match(keyword.strip()))


def _get_protected_queries(site_url: str) -> set[str]:
    """Return queries already ranking in positions 1-5 to avoid cannibalisation.

    Args:
        site_url: The GSC property URL.

    Returns:
        Set of lowercase query strings that should not be targeted again.
    """
    try:
        top_queries = gsc_tools.get_top_queries(site_url)
    except Exception:
        logger.warning("Failed to fetch GSC top queries for %s", site_url, exc_info=True)
        return set()

    protected: set[str] = set()
    for row in top_queries:
        position = row.get("position", 999)
        if position <= _CANNIBALISATION_MAX_POSITION:
            keys = row.get("keys", [])
            if keys:
                protected.add(keys[0].lower())
    return protected


def run_keyword_research(state: SEOAgentState) -> dict[str, Any]:
    """Discover keyword opportunities for the target site.

    Fetches keyword ideas from Ahrefs, filters by difficulty and volume,
    removes keywords that would cannibalise existing top-5 rankings, and
    flags People Also Ask candidates. Results are persisted to Supabase.

    Args:
        state: The current SEO agent state.

    Returns:
        State update with `keyword_opportunities`, `errors`, and `next_node`.
    """
    errors: list[str] = list(state.get("errors", []))
    opportunities: list[dict[str, Any]] = []
    target_site = state["target_site"]

    # Handle "all" — run for every site profile
    if target_site == "all":
        sites_to_run = list(SITE_PROFILES.keys())
    else:
        sites_to_run = [target_site]

    for site_key in sites_to_run:
        profile = SITE_PROFILES.get(site_key)
        if profile is None:
            msg = f"No site profile found for '{site_key}'"
            logger.error(msg)
            errors.append(msg)
            continue
        site_opps = _run_keyword_research_for_site(
            site_key, profile, state, errors
        )
        opportunities.extend(site_opps)

    logger.info(
        "Keyword research complete: %d opportunities across %s",
        len(opportunities),
        sites_to_run,
    )

    return {
        "keyword_opportunities": opportunities,
        "errors": errors,
        "next_node": "END",
    }


def _run_keyword_research_for_site(
    target_site: str,
    profile: dict,
    state: SEOAgentState,
    errors: list[str],
) -> list[dict[str, Any]]:
    """Run keyword research for a single site profile."""
    opportunities: list[dict[str, Any]] = []

    # Determine seed keywords — use the explicit seed or fall back to profile
    seed_keyword = state.get("seed_keyword")
    seed_keywords: list[str] = (
        [seed_keyword] if seed_keyword else profile.get("seed_keywords", [])
    )

    if not seed_keywords:
        msg = f"No seed keywords available for '{target_site}'"
        logger.error(msg)
        errors.append(msg)
        return {
            "keyword_opportunities": [],
            "errors": errors,
            "next_node": "END",
        }

    # Fetch protected queries from GSC to avoid cannibalisation
    gsc_property = profile.get("gsc_property", "")
    protected = _get_protected_queries(gsc_property) if gsc_property else set()
    logger.info(
        "Protected queries (positions 1-5) for %s: %d keywords",
        target_site,
        len(protected),
    )

    # Gather keyword ideas for each seed
    for seed in seed_keywords:
        try:
            ideas = ahrefs_tools.get_keyword_ideas.invoke(seed)
        except Exception:
            msg = f"Ahrefs get_keyword_ideas failed for seed '{seed}'"
            logger.warning(msg, exc_info=True)
            errors.append(msg)
            continue

        for kw in ideas:
            keyword_text = kw.get("keyword", "").lower()
            volume = kw.get("volume", 0) or 0
            # Ahrefs v3 field names vary: keywords-explorer uses "difficulty",
            # site-explorer uses "keyword_difficulty"
            kd = kw.get("difficulty") or kw.get("keyword_difficulty") or kw.get("kd") or 0

            # Apply filtering thresholds
            if kd >= _MAX_KD:
                continue
            if volume < _MIN_VOLUME:
                continue

            # Skip keywords we already rank well for
            if keyword_text in protected:
                logger.debug("Skipping protected keyword: %s", keyword_text)
                continue

            is_paa = _is_paa_keyword(keyword_text)

            opportunity: dict[str, Any] = {
                "keyword": keyword_text,
                "volume": volume,
                "kd": kd,
                "cpc": kw.get("cpc", 0.0),
                "intent": kw.get("intent", "unknown"),
                "is_paa": is_paa,
                "target_site": target_site,
                "seed_keyword": seed,
            }
            opportunities.append(opportunity)

            # Persist to Supabase
            try:
                supabase_tools.insert_record(
                    "seo_keyword_opportunities",
                    {
                        "keyword": keyword_text,
                        "volume": volume,
                        "kd": kd,
                        "cpc": kw.get("cpc", 0.0),
                        "intent": kw.get("intent", "unknown"),
                        "target_site": target_site,
                        "paa_keywords": [keyword_text] if is_paa else [],
                    },
                )
            except Exception:
                msg = f"Failed to save keyword '{keyword_text}' to Supabase"
                logger.warning(msg, exc_info=True)
                errors.append(msg)

    # Log PAA keywords separately for downstream use
    paa_keywords = [opp for opp in opportunities if opp.get("is_paa")]
    logger.info(
        "Keyword research for %s: %d opportunities (%d PAA)",
        target_site,
        len(opportunities),
        len(paa_keywords),
    )

    return opportunities
