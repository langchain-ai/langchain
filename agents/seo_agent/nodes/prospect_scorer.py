"""Prospect scorer node — assigns a 0-100 quality score and tier to each prospect.

Uses a deterministic rubric covering domain rating, traffic, competitor
overlap, discovery method, contact quality, recency, and geo signals.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from agents.seo_agent.config import MIN_OUTREACH_SCORE, TIER1_OUTREACH_SCORE
from agents.seo_agent.state import SEOAgentState
from agents.seo_agent.tools import supabase_tools

logger = logging.getLogger(__name__)

# Generic email prefixes that score lower for personalisation
_GENERIC_EMAIL_PREFIXES = frozenset({
    "info@",
    "hello@",
    "contact@",
    "admin@",
    "support@",
    "webmaster@",
    "enquiries@",
    "team@",
    "mail@",
    "office@",
})

# UK-based TLDs for the geo bonus
_UK_TLDS = (".co.uk", ".org.uk", ".uk")


def _is_generic_email(email: str) -> bool:
    """Check whether an email address uses a generic prefix.

    Args:
        email: The email address or pattern to check.

    Returns:
        True if the email starts with a known generic prefix.
    """
    email_lower = email.lower().strip()
    return any(email_lower.startswith(prefix) for prefix in _GENERIC_EMAIL_PREFIXES)


def _is_uk_domain(domain: str) -> bool:
    """Check whether a domain uses a UK-based TLD.

    Args:
        domain: The domain name to check.

    Returns:
        True if the domain ends with a UK TLD suffix.
    """
    domain_lower = domain.lower()
    return any(domain_lower.endswith(tld) for tld in _UK_TLDS)


def _is_recently_updated(prospect: dict[str, Any]) -> bool:
    """Check whether the prospect page was updated in the last 12 months.

    Args:
        prospect: The prospect record dict.

    Returns:
        True if the page has a `published` or `updated_at` date within 12 months.
    """
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=365)
    for field in ("published", "updated_at", "last_seen"):
        date_str = prospect.get(field, "")
        if not date_str:
            continue
        try:
            dt = datetime.fromisoformat(str(date_str))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt >= cutoff:
                return True
        except (ValueError, TypeError):
            continue
    return False


def _score_prospect(prospect: dict[str, Any]) -> tuple[int, list[str]]:
    """Calculate the quality score for a single prospect.

    Scoring rubric (0-100):
        - DR 60+: 25 pts | DR 40-59: 15 pts | DR 20-39: 8 pts
        - Real traffic 1k+/mo: 15 pts
        - Links to 2+ competitors: 20 pts
        - Unlinked brand mention: 25 pts
        - Broken link opportunity: 20 pts
        - Resource page: 15 pts
        - Contact email found (not generic): 10 pts
        - Page updated in last 12 months: 10 pts
        - UK-based domain: 10 pts

    Args:
        prospect: The enriched prospect record.

    Returns:
        Tuple of (score, list of reasons for the score breakdown).
    """
    score = 0
    reasons: list[str] = []

    # Domain Rating
    dr = prospect.get("dr", 0) or 0
    if dr >= 60:
        score += 25
        reasons.append(f"DR {dr} (60+): +25")
    elif dr >= 40:
        score += 15
        reasons.append(f"DR {dr} (40-59): +15")
    elif dr >= 20:
        score += 8
        reasons.append(f"DR {dr} (20-39): +8")

    # Traffic
    traffic = prospect.get("monthly_traffic", 0) or 0
    if traffic >= 1000:
        score += 15
        reasons.append(f"Traffic {traffic}/mo (1k+): +15")

    # Competitor links
    competitor_names = prospect.get("competitor_names", []) or []
    if len(competitor_names) >= 2:
        score += 20
        reasons.append(f"Links to {len(competitor_names)} competitors: +20")

    # Discovery method bonuses
    method = prospect.get("discovery_method", "")

    if method == "unlinked_mention":
        score += 25
        reasons.append("Unlinked brand mention: +25")

    if method == "broken_link":
        score += 20
        reasons.append("Broken link opportunity: +20")

    if method == "resource_page":
        score += 15
        reasons.append("Resource page: +15")

    # Contact email quality
    contact_email = prospect.get("contact_email", "") or ""
    if contact_email and not _is_generic_email(contact_email):
        score += 10
        reasons.append("Non-generic contact email found: +10")

    # Recency
    if _is_recently_updated(prospect):
        score += 10
        reasons.append("Page updated within 12 months: +10")

    # UK geo bonus
    domain = prospect.get("domain", "")
    if _is_uk_domain(domain):
        score += 10
        reasons.append("UK-based domain: +10")

    return score, reasons


def _get_enriched_prospects(state: SEOAgentState) -> list[dict[str, Any]]:
    """Get enriched prospects from state or fall back to Supabase query.

    Args:
        state: The current SEO agent state.

    Returns:
        List of enriched prospect dicts.
    """
    prospects = state.get("enriched_prospects", [])
    if prospects:
        return prospects

    return supabase_tools.query_table(
        "seo_backlink_prospects",
        filters={"status": "enriched"},
        limit=500,
        order_by="created_at",
        order_desc=False,
    )


def run_prospect_scorer(state: SEOAgentState) -> dict[str, Any]:
    """Score and tier enriched backlink prospects.

    Applies a deterministic scoring rubric (0-100) to each enriched prospect.
    Prospects scoring below the minimum threshold are rejected. Remaining
    prospects are assigned to tier 1 or tier 2 based on their score.

    Args:
        state: The current SEO agent state.

    Returns:
        State update with `scored_prospects`, `errors`, and `next_node`.
    """
    errors: list[str] = list(state.get("errors", []))
    target_site = state["target_site"]
    scored: list[dict[str, Any]] = []

    prospects = _get_enriched_prospects(state)
    if not prospects:
        logger.info("No enriched prospects to score for %s", target_site)
        return {
            "scored_prospects": [],
            "errors": errors,
            "next_node": "END",
        }

    logger.info("Scoring %d enriched prospects for %s", len(prospects), target_site)

    rejected_count = 0
    tier1_count = 0
    tier2_count = 0

    for prospect in prospects:
        prospect_id = prospect.get("id", "")
        domain = prospect.get("domain", "")

        try:
            score, reasons = _score_prospect(prospect)
        except Exception:
            msg = f"Failed to score prospect {domain}"
            logger.warning(msg, exc_info=True)
            errors.append(msg)
            continue

        # Determine tier and status
        if score < MIN_OUTREACH_SCORE:
            tier = None
            status = "rejected"
            reject_reason = (
                f"Score {score} below minimum {MIN_OUTREACH_SCORE}: "
                + "; ".join(reasons) if reasons else "no qualifying signals"
            )
            logger.debug(
                "Rejected prospect %s (score %d): %s",
                domain,
                score,
                reject_reason,
            )
            rejected_count += 1
        elif score >= TIER1_OUTREACH_SCORE:
            tier = "tier1"
            status = "scored"
            tier1_count += 1
        else:
            tier = "tier2"
            status = "scored"
            tier2_count += 1

        # Build update record
        update_data: dict[str, Any] = {
            "score": score,
            "tier": tier,
            "status": status,
        }

        if prospect_id:
            update_data["id"] = prospect_id

        # Persist to Supabase
        try:
            if prospect_id:
                updated = supabase_tools.upsert_record(
                    "seo_backlink_prospects", update_data
                )
            else:
                updated = {**prospect, **update_data}
            scored.append(updated)
        except Exception:
            msg = f"Failed to update score for prospect {domain}"
            logger.warning(msg, exc_info=True)
            errors.append(msg)
            scored.append({**prospect, **update_data})

    logger.info(
        "Scoring complete for %s: %d tier1, %d tier2, %d rejected",
        target_site,
        tier1_count,
        tier2_count,
        rejected_count,
    )

    return {
        "scored_prospects": scored,
        "errors": errors,
        "next_node": "END",
    }
