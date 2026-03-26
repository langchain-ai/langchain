"""CRM & data layer — Ralf's working memory for keywords, rankings, prospects, and content.

This module is the brain between the APIs and the database. Instead of hitting
Ahrefs for every request, Ralf checks the local cache first. Instead of treating
prospects as a flat list, Ralf tracks communication stages like a CRM.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import Any

from agents.seo_agent.tools.supabase_tools import (
    get_client,
    insert_record,
    query_table,
    upsert_record,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Keyword cache — avoid redundant Ahrefs calls
# ---------------------------------------------------------------------------

_KEYWORD_CACHE_DAYS = 7  # Re-fetch from Ahrefs if older than this


def get_cached_keyword(keyword: str, country: str = "gb") -> dict | None:
    """Check the keyword cache before calling Ahrefs.

    Returns cached data if fresh enough, or None if a refresh is needed.
    """
    rows = query_table(
        "seo_keyword_cache",
        filters={"keyword": keyword.lower(), "country": country},
        limit=1,
    )
    if not rows:
        return None

    row = rows[0]
    last_updated = row.get("last_updated", "")
    if isinstance(last_updated, str) and last_updated:
        try:
            updated_dt = datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
            if (datetime.now(tz=timezone.utc) - updated_dt).days <= _KEYWORD_CACHE_DAYS:
                return row
        except (ValueError, TypeError):
            pass
    return None


def cache_keyword(keyword: str, country: str, data: dict) -> None:
    """Save keyword data to the cache."""
    try:
        upsert_record(
            "seo_keyword_cache",
            {
                "keyword": keyword.lower(),
                "country": country,
                "volume": data.get("volume"),
                "difficulty": data.get("difficulty") or data.get("keyword_difficulty"),
                "cpc": data.get("cpc"),
                "traffic_potential": data.get("traffic_potential"),
                "intent": data.get("intent"),
                "last_updated": datetime.now(tz=timezone.utc).isoformat(),
                "source": "ahrefs",
            },
            on_conflict="keyword,country",
        )
    except Exception:
        logger.warning("Failed to cache keyword %s", keyword, exc_info=True)


def get_all_cached_keywords(country: str = "gb", limit: int = 200) -> list[dict]:
    """Get all cached keywords, sorted by opportunity score."""
    rows = query_table("seo_keyword_cache", limit=limit)
    return sorted(
        [r for r in rows if r.get("country") == country],
        key=lambda r: (r.get("volume", 0) or 0) / ((r.get("difficulty", 50) or 50) + 1),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Rank tracking — snapshot our positions and competitors
# ---------------------------------------------------------------------------


def snapshot_our_rankings(target_site: str, rankings: list[dict]) -> int:
    """Save a batch of our ranking positions to the database.

    Returns the number of records saved.
    """
    today = date.today().isoformat()
    saved = 0
    for r in rankings:
        try:
            # Get previous position for change calculation
            prev_rows = query_table(
                "seo_our_rankings",
                filters={"keyword": r.get("keyword", "").lower(), "target_site": target_site},
                limit=1,
                order_by="snapshot_date",
                order_desc=True,
            )
            prev_pos = prev_rows[0].get("position") if prev_rows else None
            current_pos = r.get("position") or r.get("best_position")

            change = None
            if prev_pos is not None and current_pos is not None:
                change = prev_pos - current_pos  # positive = improved

            insert_record("seo_our_rankings", {
                "target_site": target_site,
                "keyword": r.get("keyword", "").lower(),
                "position": current_pos,
                "url": r.get("url", ""),
                "previous_position": prev_pos,
                "change": change,
                "volume": r.get("volume"),
                "snapshot_date": today,
            })
            saved += 1
        except Exception:
            logger.warning("Failed to save ranking for %s", r.get("keyword"), exc_info=True)
    return saved


def snapshot_competitor_rankings(
    target_site: str, competitor_domain: str, keywords: list[dict]
) -> int:
    """Save competitor ranking data."""
    today = date.today().isoformat()
    saved = 0
    for kw in keywords:
        try:
            insert_record("seo_competitor_rankings", {
                "competitor_domain": competitor_domain,
                "keyword": kw.get("keyword", "").lower(),
                "position": kw.get("position"),
                "volume": kw.get("volume"),
                "traffic": kw.get("sum_traffic") or kw.get("traffic"),
                "target_site": target_site,
                "snapshot_date": today,
            })
            saved += 1
        except Exception:
            logger.warning("Failed to save competitor ranking", exc_info=True)
    return saved


def get_ranking_history(keyword: str, target_site: str, days: int = 90) -> list[dict]:
    """Get ranking history for a keyword over time."""
    rows = query_table(
        "seo_our_rankings",
        filters={"keyword": keyword.lower(), "target_site": target_site},
        limit=days,
        order_by="snapshot_date",
        order_desc=True,
    )
    return rows


def get_ranking_movers(target_site: str, limit: int = 10) -> dict:
    """Get keywords with biggest position changes (winners and losers)."""
    all_rankings = query_table(
        "seo_our_rankings",
        filters={"target_site": target_site},
        limit=200,
        order_by="snapshot_date",
        order_desc=True,
    )
    # Group by keyword, get latest
    latest: dict[str, dict] = {}
    for r in all_rankings:
        kw = r.get("keyword", "")
        if kw not in latest:
            latest[kw] = r

    with_changes = [r for r in latest.values() if r.get("change") is not None]
    winners = sorted(with_changes, key=lambda r: r.get("change", 0), reverse=True)[:limit]
    losers = sorted(with_changes, key=lambda r: r.get("change", 0))[:limit]

    return {"winners": winners, "losers": losers}


# ---------------------------------------------------------------------------
# Prospect CRM — track communication with outreach targets
# ---------------------------------------------------------------------------

PROSPECT_STAGES = [
    "new",           # Just discovered
    "enriched",      # Contact info found
    "scored",        # Scored and tiered
    "email_drafted", # Outreach email generated
    "contacted",     # Initial email sent
    "followed_up",   # Follow-up sent
    "replied",       # Got a reply
    "negotiating",   # In discussion about link
    "link_acquired", # Backlink secured
    "rejected",      # They said no or didn't respond after follow-ups
    "blocked",       # On the blocklist
]


def log_prospect_communication(
    prospect_id: str,
    domain: str,
    direction: str = "outbound",
    channel: str = "email",
    subject: str = "",
    body_preview: str = "",
    contact_name: str = "",
    contact_email: str = "",
    status: str = "sent",
    notes: str = "",
) -> dict:
    """Log a communication event with a prospect."""
    record = {
        "prospect_id": prospect_id,
        "domain": domain,
        "contact_name": contact_name,
        "contact_email": contact_email,
        "channel": channel,
        "direction": direction,
        "subject": subject,
        "body_preview": body_preview[:500] if body_preview else "",
        "status": status,
        "sent_at": datetime.now(tz=timezone.utc).isoformat() if direction == "outbound" else None,
        "notes": notes,
    }
    return insert_record("seo_prospect_communications", record)


def update_prospect_stage(prospect_id: str, new_status: str, notes: str = "") -> dict:
    """Update a prospect's pipeline stage."""
    update_data: dict[str, Any] = {"status": new_status}
    if new_status == "contacted":
        update_data["last_contacted_at"] = datetime.now(tz=timezone.utc).isoformat()
    if new_status == "followed_up":
        update_data["follow_up_count"] = 1  # Will need increment logic
    if new_status == "replied":
        update_data["reply_received"] = True

    return upsert_record("seo_backlink_prospects", {
        "id": prospect_id,
        **update_data,
    })


def get_prospect_pipeline() -> dict[str, list[dict]]:
    """Get all prospects grouped by pipeline stage."""
    all_prospects = query_table("seo_backlink_prospects", limit=500)
    pipeline: dict[str, list[dict]] = {stage: [] for stage in PROSPECT_STAGES}
    for p in all_prospects:
        stage = p.get("status", "new")
        if stage in pipeline:
            pipeline[stage].append(p)
        else:
            pipeline["new"].append(p)
    return pipeline


def get_prospects_needing_followup(days_since_contact: int = 7) -> list[dict]:
    """Find prospects that were contacted but haven't replied and need follow-up."""
    cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=days_since_contact)).isoformat()
    contacted = query_table(
        "seo_backlink_prospects",
        filters={"status": "contacted"},
        limit=100,
    )
    needs_followup = []
    for p in contacted:
        last_contact = p.get("last_contacted_at", "")
        if last_contact and last_contact < cutoff and not p.get("reply_received"):
            needs_followup.append(p)
    return needs_followup


def get_communication_history(domain: str) -> list[dict]:
    """Get all communications with a specific domain."""
    return query_table(
        "seo_prospect_communications",
        filters={"domain": domain},
        limit=50,
        order_by="created_at",
        order_desc=True,
    )


# ---------------------------------------------------------------------------
# Content performance — track which posts drive traffic
# ---------------------------------------------------------------------------


def snapshot_content_performance(target_site: str, pages: list[dict]) -> int:
    """Save content performance data for a site's pages."""
    today = date.today().isoformat()
    saved = 0
    for page in pages:
        try:
            insert_record("seo_content_performance", {
                "url": page.get("url", ""),
                "title": page.get("title") or page.get("top_keyword", ""),
                "target_site": target_site,
                "target_keyword": page.get("top_keyword", ""),
                "organic_traffic": page.get("sum_traffic") or page.get("traffic", 0),
                "keywords_ranking": page.get("keywords", 0),
                "best_position": page.get("position"),
                "snapshot_date": today,
            })
            saved += 1
        except Exception:
            logger.warning("Failed to save content performance", exc_info=True)
    return saved


def get_content_performance_history(url: str, days: int = 90) -> list[dict]:
    """Get performance history for a specific URL."""
    return query_table(
        "seo_content_performance",
        filters={"url": url},
        limit=days,
        order_by="snapshot_date",
        order_desc=True,
    )


def add_tracked_keyword(target_site: str, keyword: str, target_url: str = "") -> None:
    """Add a keyword to our active tracking watchlist after publishing content for it."""
    try:
        upsert_record(
            "seo_our_rankings",
            {
                "target_site": target_site,
                "keyword": keyword.lower(),
                "url": target_url,
                "position": None,  # Unknown until first snapshot
                "snapshot_date": date.today().isoformat(),
                "notes": "auto-tracked after content publish",
            },
            on_conflict="keyword,target_site,snapshot_date",
        )
    except Exception:
        logger.warning("Failed to add tracked keyword: %s", keyword, exc_info=True)


def get_top_performing_content(target_site: str, limit: int = 20) -> list[dict]:
    """Get top-performing content by traffic."""
    rows = query_table(
        "seo_content_performance",
        filters={"target_site": target_site},
        limit=limit,
        order_by="organic_traffic",
        order_desc=True,
    )
    # Deduplicate by URL (keep latest)
    seen: set[str] = set()
    unique: list[dict] = []
    for r in rows:
        url = r.get("url", "")
        if url not in seen:
            seen.add(url)
            unique.append(r)
    return unique


# ---------------------------------------------------------------------------
# Unified dashboard — everything Ralf needs to know at a glance
# ---------------------------------------------------------------------------


def get_dashboard_summary() -> dict[str, Any]:
    """Generate a complete dashboard summary of the SEO operation."""
    from agents.seo_agent.tools.supabase_tools import get_weekly_spend

    keywords = query_table("seo_keyword_opportunities", limit=500)
    cached_kw = query_table("seo_keyword_cache", limit=500)
    gaps = query_table("seo_content_gaps", limit=500)
    briefs = query_table("seo_content_briefs", limit=500)
    prospects = query_table("seo_backlink_prospects", limit=500)
    rankings = query_table("seo_our_rankings", limit=500)
    spend = get_weekly_spend()

    pipeline = get_prospect_pipeline()
    pipeline_summary = {
        stage: len(prospects_in_stage)
        for stage, prospects_in_stage in pipeline.items()
        if prospects_in_stage
    }

    return {
        "keywords_discovered": len(keywords),
        "keywords_cached": len(cached_kw),
        "content_gaps": len(gaps),
        "content_pieces": len(briefs),
        "prospects_total": len(prospects),
        "prospect_pipeline": pipeline_summary,
        "rankings_tracked": len(set(r.get("keyword", "") for r in rankings)),
        "weekly_spend": spend,
        "sites": {
            "freeroomplanner": {
                "keywords": len([k for k in keywords if k.get("target_site") == "freeroomplanner"]),
                "content": len([b for b in briefs if b.get("target_site") == "freeroomplanner"]),
                "prospects": len([p for p in prospects if p.get("target_site") == "freeroomplanner"]),
            },
            "kitchensdirectory": {
                "keywords": len([k for k in keywords if k.get("target_site") == "kitchensdirectory"]),
                "content": len([b for b in briefs if b.get("target_site") == "kitchensdirectory"]),
                "prospects": len([p for p in prospects if p.get("target_site") == "kitchensdirectory"]),
            },
            "kitchen_estimator": {
                "keywords": len([k for k in keywords if k.get("target_site") == "kitchen_estimator"]),
                "content": len([b for b in briefs if b.get("target_site") == "kitchen_estimator"]),
                "prospects": len([p for p in prospects if p.get("target_site") == "kitchen_estimator"]),
            },
        },
    }
