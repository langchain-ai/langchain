"""Rank tracker node — monitors keyword rankings and detects movements.

Pulls weekly data from Google Search Console and Ahrefs rank tracking,
flags significant position changes, and detects URL cannibalisation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from agents.seo_agent.config import SITE_PROFILES
from agents.seo_agent.state import SEOAgentState
from agents.seo_agent.tools import ahrefs_tools, gsc_tools, supabase_tools

logger = logging.getLogger(__name__)

# A keyword that moved 3+ positions in either direction is flagged
_SIGNIFICANT_MOVEMENT_THRESHOLD = 3


def _detect_cannibalisation(
    rank_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect keyword cannibalisation — multiple URLs ranking for the same keyword.

    Args:
        rank_entries: List of rank tracking dicts with `keyword` and `url` keys.

    Returns:
        List of cannibalisation issue dicts.
    """
    keyword_urls: dict[str, list[str]] = {}
    for entry in rank_entries:
        kw = entry.get("keyword", "")
        url = entry.get("url", "")
        if kw and url:
            keyword_urls.setdefault(kw, []).append(url)

    issues: list[dict[str, Any]] = []
    for kw, urls in keyword_urls.items():
        unique_urls = list(set(urls))
        if len(unique_urls) > 1:
            issues.append({
                "keyword": kw,
                "urls": unique_urls,
                "url_count": len(unique_urls),
                "issue": "cannibalisation",
            })

    return issues


def _flag_significant_movements(
    rank_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flag keywords that moved 3+ positions in either direction.

    Args:
        rank_entries: List of rank tracking dicts with `position` and
            `previous_position` keys.

    Returns:
        List of flagged movement dicts.
    """
    flagged: list[dict[str, Any]] = []
    for entry in rank_entries:
        position = entry.get("position")
        previous = entry.get("previous_position")
        if position is None or previous is None:
            continue

        movement = previous - position  # positive = improved
        if abs(movement) >= _SIGNIFICANT_MOVEMENT_THRESHOLD:
            direction = "improved" if movement > 0 else "declined"
            flagged.append({
                "keyword": entry.get("keyword", ""),
                "url": entry.get("url", ""),
                "position": position,
                "previous_position": previous,
                "movement": movement,
                "direction": direction,
            })

    return flagged


def run_rank_tracker(state: SEOAgentState) -> dict[str, Any]:
    """Track keyword rankings across sites and flag movements.

    Pulls weekly search analytics from GSC and rank tracking data from
    Ahrefs. Flags keywords with significant position changes and detects
    URL cannibalisation. Saves all results to the `seo_rank_history` table.

    Args:
        state: The current SEO agent state.

    Returns:
        State update with `rank_data`, `errors`, and `next_node`.
    """
    errors: list[str] = list(state.get("errors", []))
    all_rank_data: list[dict[str, Any]] = []
    target_site = state.get("target_site", "")

    # Determine which sites to track
    if target_site and target_site in SITE_PROFILES:
        sites_to_track = {target_site: SITE_PROFILES[target_site]}
    else:
        sites_to_track = SITE_PROFILES

    now = datetime.now(tz=timezone.utc)
    end_date = (now - timedelta(days=3)).strftime("%Y-%m-%d")
    start_date = (now - timedelta(days=10)).strftime("%Y-%m-%d")
    today_str = now.strftime("%Y-%m-%d")

    for site_key, profile in sites_to_track.items():
        gsc_property = profile.get("gsc_property", "")
        domain = profile.get("domain", "")

        logger.info("Tracking ranks for %s (%s)", site_key, domain)

        # Pull GSC search analytics
        gsc_rows: list[dict[str, Any]] = []
        if gsc_property:
            try:
                gsc_rows = gsc_tools.get_search_analytics(
                    site_url=gsc_property,
                    start_date=start_date,
                    end_date=end_date,
                    dimensions=["query"],
                )
            except Exception:
                msg = f"Failed to fetch GSC data for '{site_key}'"
                logger.warning(msg, exc_info=True)
                errors.append(msg)

        # Pull Ahrefs rank tracking
        ahrefs_rows: list[dict[str, Any]] = []
        if domain:
            try:
                ahrefs_rows = ahrefs_tools.get_rank_tracking.invoke(domain)
            except Exception:
                msg = f"Failed to fetch Ahrefs rank tracking for '{site_key}'"
                logger.warning(msg, exc_info=True)
                errors.append(msg)

        # Merge data — prefer Ahrefs for position data, GSC for click data
        merged: list[dict[str, Any]] = []

        # Index GSC data by keyword for merging
        gsc_by_keyword: dict[str, dict[str, Any]] = {}
        for row in gsc_rows:
            keys = row.get("keys", [])
            if keys:
                gsc_by_keyword[keys[0].lower()] = row

        for ahrefs_entry in ahrefs_rows:
            kw = ahrefs_entry.get("keyword", "").lower()
            gsc_entry = gsc_by_keyword.pop(kw, {})

            record: dict[str, Any] = {
                "date": today_str,
                "keyword": kw,
                "url": ahrefs_entry.get("url", ""),
                "position": ahrefs_entry.get("position"),
                "previous_position": ahrefs_entry.get("previous_position"),
                "impressions": gsc_entry.get("impressions", 0),
                "clicks": gsc_entry.get("clicks", 0),
                "target_site": site_key,
                "volume": ahrefs_entry.get("volume", 0),
            }
            merged.append(record)

        # Add remaining GSC-only entries
        for kw, gsc_entry in gsc_by_keyword.items():
            record = {
                "date": today_str,
                "keyword": kw,
                "url": "",
                "position": gsc_entry.get("position"),
                "previous_position": None,
                "impressions": gsc_entry.get("impressions", 0),
                "clicks": gsc_entry.get("clicks", 0),
                "target_site": site_key,
                "volume": 0,
            }
            merged.append(record)

        # Flag significant movements
        movements = _flag_significant_movements(merged)
        for movement in movements:
            logger.info(
                "Rank %s: '%s' moved %+d positions (now #%s)",
                movement["direction"],
                movement["keyword"],
                movement["movement"],
                movement["position"],
            )

        # Detect cannibalisation
        cannibalisation_issues = _detect_cannibalisation(merged)
        for issue in cannibalisation_issues:
            logger.warning(
                "Cannibalisation detected for '%s': %d URLs ranking (%s)",
                issue["keyword"],
                issue["url_count"],
                ", ".join(issue["urls"]),
            )
            merged.append({
                "date": today_str,
                "keyword": issue["keyword"],
                "url": ", ".join(issue["urls"]),
                "position": None,
                "previous_position": None,
                "impressions": 0,
                "clicks": 0,
                "target_site": site_key,
                "issue": "cannibalisation",
            })

        # Persist to Supabase
        for record in merged:
            try:
                supabase_tools.insert_record(
                    "seo_rank_history",
                    {
                        "date": record.get("date", today_str),
                        "keyword": record.get("keyword", ""),
                        "url": record.get("url", ""),
                        "position": record.get("position"),
                        "previous_position": record.get("previous_position"),
                        "impressions": record.get("impressions", 0),
                        "clicks": record.get("clicks", 0),
                        "target_site": site_key,
                    },
                )
            except Exception:
                msg = (
                    f"Failed to save rank data for "
                    f"'{record.get('keyword', '')}' to Supabase"
                )
                logger.warning(msg, exc_info=True)
                errors.append(msg)

        all_rank_data.extend(merged)

    logger.info(
        "Rank tracking complete: %d total entries across %d sites",
        len(all_rank_data),
        len(sites_to_track),
    )

    return {
        "rank_data": all_rank_data,
        "errors": errors,
        "next_node": "END",
    }
