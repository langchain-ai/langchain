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


# ---------------------------------------------------------------------------
# Outreach CRM — kitchen/bathroom companies, interior designers
# ---------------------------------------------------------------------------

CRM_CATEGORIES: list[str] = [
    "kitchen_company",
    "bathroom_company",
    "interior_designer",
    "blogger",
]

CRM_SUBCATEGORIES: dict[str, list[str]] = {
    "kitchen_company": ["showroom", "manufacturer", "fitter", "supplier", "bespoke_maker"],
    "bathroom_company": ["showroom", "fitter", "supplier", "designer"],
    "interior_designer": ["freelance", "studio", "firm"],
    "blogger": ["food", "lifestyle", "home_improvement", "diy"],
}

CRM_OUTREACH_STATUSES: list[str] = [
    "not_contacted",
    "contacted",
    "replied",
    "partnership_active",
    "declined",
    "blocked",
]


def add_crm_contact(
    company_name: str,
    category: str,
    *,
    contact_name: str = "",
    contact_role: str = "",
    email: str = "",
    phone: str = "",
    website: str = "",
    city: str = "",
    region: str = "",
    postcode: str = "",
    country: str = "GB",
    subcategory: str = "",
    instagram: str = "",
    facebook: str = "",
    linkedin: str = "",
    outreach_segment: str = "",
    score: int = 0,
    tier: str = "",
    backlink_prospect_id: str = "",
    source: str = "manual",
    tags: list[str] | None = None,
    notes: str = "",
) -> dict:
    """Add a new contact to the outreach CRM.

    Args:
        company_name: Company or business name.
        category: One of ``CRM_CATEGORIES`` (kitchen_company, bathroom_company,
            interior_designer).
        contact_name: Name of the primary contact person.
        contact_role: Role/title of the contact.
        email: Email address.
        phone: Phone number.
        website: Company website URL.
        city: City where the company is based.
        region: Region or county.
        postcode: Postal code.
        country: ISO country code.
        subcategory: Finer classification within the category.
        instagram: Instagram handle or URL.
        facebook: Facebook page URL.
        linkedin: LinkedIn profile or company page URL.
        outreach_segment: Key from ``OUTREACH_SEGMENTS`` in outreach_strategy.
        score: Outreach priority score.
        tier: Tier classification (tier_1, tier_2, tier_3).
        backlink_prospect_id: Optional link to ``seo_backlink_prospects``.
        source: How this contact was found.
        tags: Flexible tags for categorization.
        notes: Free-text notes.

    Returns:
        The inserted record dict.
    """
    now = datetime.now(tz=timezone.utc).isoformat()
    record: dict[str, Any] = {
        "company_name": company_name,
        "category": category,
        "contact_name": contact_name,
        "contact_role": contact_role,
        "email": email,
        "phone": phone,
        "website": website,
        "city": city,
        "region": region,
        "postcode": postcode,
        "country": country,
        "subcategory": subcategory,
        "instagram": instagram,
        "facebook": facebook,
        "linkedin": linkedin,
        "outreach_status": "not_contacted",
        "outreach_segment": outreach_segment,
        "score": score,
        "tier": tier,
        "source": source,
        "tags": tags or [],
        "notes": notes,
        "created_at": now,
        "updated_at": now,
    }
    if backlink_prospect_id:
        record["backlink_prospect_id"] = backlink_prospect_id
    return insert_record("crm_contacts", record)


def update_crm_contact(contact_id: str, **fields: Any) -> dict:
    """Update fields on an existing CRM contact.

    Args:
        contact_id: UUID of the contact to update.
        **fields: Column names and their new values.

    Returns:
        The updated record dict.
    """
    fields["updated_at"] = datetime.now(tz=timezone.utc).isoformat()
    return upsert_record("crm_contacts", {"id": contact_id, **fields})


def get_crm_contacts(
    *,
    category: str | None = None,
    city: str | None = None,
    outreach_status: str | None = None,
    outreach_segment: str | None = None,
    limit: int = 100,
) -> list[dict]:
    """Query CRM contacts with optional filters.

    Args:
        category: Filter by category (kitchen_company, bathroom_company,
            interior_designer).
        city: Filter by city.
        outreach_status: Filter by outreach status.
        outreach_segment: Filter by outreach segment.
        limit: Maximum rows to return.

    Returns:
        List of matching contact dicts.
    """
    filters: dict[str, Any] = {}
    if category:
        filters["category"] = category
    if city:
        filters["city"] = city
    if outreach_status:
        filters["outreach_status"] = outreach_status
    if outreach_segment:
        filters["outreach_segment"] = outreach_segment
    return query_table("crm_contacts", filters=filters, limit=limit)


def search_crm_contacts(query_text: str, *, limit: int = 50) -> list[dict]:
    """Search CRM contacts by company name or city.

    Args:
        query_text: Text to search for (case-insensitive partial match).
        limit: Maximum rows to return.

    Returns:
        List of matching contact dicts.
    """
    if os.getenv("SUPABASE_MOCK", "false").lower() in ("true", "1", "yes"):
        from agents.seo_agent.tools.supabase_tools import _mock_store

        rows = _mock_store.get("crm_contacts", [])
        q = query_text.lower()
        return [
            r for r in rows
            if q in (r.get("company_name", "") or "").lower()
            or q in (r.get("city", "") or "").lower()
        ][:limit]

    client = get_client()
    resp = (
        client.table("crm_contacts")
        .select("*")
        .ilike("company_name", f"%{query_text}%")
        .limit(limit)
        .execute()
    )
    return resp.data or []


def log_crm_interaction(
    contact_id: str,
    interaction_type: str,
    *,
    direction: str = "outbound",
    channel: str = "email",
    subject: str = "",
    body_preview: str = "",
    performed_by: str = "ralf",
) -> dict:
    """Log an interaction with a CRM contact.

    Args:
        contact_id: UUID of the contact.
        interaction_type: Type of interaction (email_sent, email_received,
            phone_call, meeting, note, social_dm).
        direction: Direction of the interaction (outbound, inbound, internal).
        channel: Communication channel.
        subject: Subject line or title.
        body_preview: Short preview of the message body.
        performed_by: Agent or person who performed this action.

    Returns:
        The inserted interaction record.
    """
    return insert_record("crm_interactions", {
        "contact_id": contact_id,
        "interaction_type": interaction_type,
        "direction": direction,
        "channel": channel,
        "subject": subject,
        "body_preview": body_preview[:500] if body_preview else "",
        "status": "logged",
        "performed_by": performed_by,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
    })


def get_crm_interaction_history(contact_id: str, *, limit: int = 50) -> list[dict]:
    """Get all interactions for a CRM contact.

    Args:
        contact_id: UUID of the contact.
        limit: Maximum rows to return.

    Returns:
        List of interaction dicts, newest first.
    """
    return query_table(
        "crm_interactions",
        filters={"contact_id": contact_id},
        limit=limit,
        order_by="created_at",
        order_desc=True,
    )


def update_crm_outreach_status(
    contact_id: str, new_status: str, *, notes: str = ""
) -> dict:
    """Update a CRM contact's outreach status.

    Args:
        contact_id: UUID of the contact.
        new_status: New status (must be in ``CRM_OUTREACH_STATUSES``).
        notes: Optional note to log alongside the status change.

    Returns:
        The updated contact record.
    """
    if new_status not in CRM_OUTREACH_STATUSES:
        msg = f"Invalid CRM outreach status: {new_status!r}. Must be one of {CRM_OUTREACH_STATUSES}"
        raise ValueError(msg)

    update_data: dict[str, Any] = {
        "outreach_status": new_status,
        "updated_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    if new_status == "contacted":
        update_data["last_contacted_at"] = datetime.now(tz=timezone.utc).isoformat()

    result = upsert_record("crm_contacts", {"id": contact_id, **update_data})

    if notes:
        log_crm_interaction(
            contact_id,
            "note",
            direction="internal",
            channel="system",
            subject=f"Status changed to {new_status}",
            body_preview=notes,
        )

    return result


def get_crm_pipeline() -> dict[str, list[dict]]:
    """Group all CRM contacts by outreach status.

    Returns:
        Dict mapping each status to its list of contacts.
    """
    all_contacts = query_table("crm_contacts", limit=1000)
    pipeline: dict[str, list[dict]] = {status: [] for status in CRM_OUTREACH_STATUSES}
    for c in all_contacts:
        status = c.get("outreach_status", "not_contacted")
        if status in pipeline:
            pipeline[status].append(c)
        else:
            pipeline["not_contacted"].append(c)
    return pipeline


def import_kitchen_makers_to_crm(city: str | None = None) -> int:
    """Import kitchen makers from the ``kitchen_makers`` table into the CRM.

    Args:
        city: Optional city filter. Imports all makers if None.

    Returns:
        Number of contacts imported.
    """
    from agents.seo_agent.tools.supabase_tools import get_makers_by_location

    makers = get_makers_by_location(city or "")
    imported = 0
    for maker in makers:
        try:
            add_crm_contact(
                company_name=maker.get("name") or "Unknown",
                category="kitchen_company",
                subcategory="bespoke_maker",
                email=maker.get("email") or "",
                phone=maker.get("phone") or "",
                website=maker.get("website") or "",
                city=maker.get("city") or "",
                region=maker.get("region") or "",
                postcode=maker.get("postcode") or "",
                country=maker.get("country") or "GB",
                source="kitchen_makers_import",
                notes=maker.get("description") or "",
            )
            imported += 1
        except Exception:
            logger.warning(
                "Failed to import kitchen maker: %s", maker.get("name"), exc_info=True
            )
    return imported


def get_crm_contacts_needing_followup(days_since_contact: int = 7) -> list[dict]:
    """Find CRM contacts that were contacted but need a follow-up.

    Args:
        days_since_contact: Number of days since last contact before
            a follow-up is due.

    Returns:
        List of contacts needing follow-up.
    """
    cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=days_since_contact)).isoformat()
    contacted = query_table(
        "crm_contacts",
        filters={"outreach_status": "contacted"},
        limit=200,
    )
    return [
        c for c in contacted
        if c.get("last_contacted_at") and c["last_contacted_at"] < cutoff
    ]


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

    # CRM stats
    crm_contacts = query_table("crm_contacts", limit=1000)
    crm_pipeline = get_crm_pipeline()
    crm_pipeline_summary = {
        status: len(contacts)
        for status, contacts in crm_pipeline.items()
        if contacts
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
        "crm_contacts_total": len(crm_contacts),
        "crm_pipeline": crm_pipeline_summary,
        "crm_by_category": {
            cat: len([c for c in crm_contacts if c.get("category") == cat])
            for cat in CRM_CATEGORIES
        },
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
