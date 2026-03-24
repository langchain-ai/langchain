"""Outreach reporter node — generates a weekly markdown summary of outreach activity.

Queries Supabase for prospect discovery, scoring, email sends, replies,
and link acquisitions, then produces a structured markdown report.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any

from agents.seo_agent.state import SEOAgentState
from agents.seo_agent.tools import supabase_tools

logger = logging.getLogger(__name__)


def _get_week_boundaries() -> tuple[str, str]:
    """Calculate the ISO start and end timestamps for the current week.

    Returns:
        Tuple of (week_start_iso, week_end_iso) strings.
    """
    now = datetime.now(tz=timezone.utc)
    week_start = now - timedelta(days=now.weekday())
    week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
    week_end = week_start + timedelta(days=7)
    return week_start.isoformat(), week_end.isoformat()


def _query_week_records(
    table: str,
    date_column: str = "created_at",
    extra_filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Query records created during the current week.

    Args:
        table: The Supabase table to query.
        date_column: The timestamp column to filter on.
        extra_filters: Additional equality filters to apply.

    Returns:
        List of matching records.
    """
    week_start, _ = _get_week_boundaries()
    filters = dict(extra_filters) if extra_filters else {}

    try:
        all_records = supabase_tools.query_table(
            table,
            filters=filters if filters else None,
            limit=10000,
            order_by=date_column,
            order_desc=True,
        )
    except Exception:
        logger.warning(
            "Failed to query %s for weekly report", table, exc_info=True
        )
        return []

    # Filter to current week client-side
    results: list[dict[str, Any]] = []
    for record in all_records:
        date_str = record.get(date_column, "")
        if not date_str:
            continue
        try:
            if str(date_str) >= week_start:
                results.append(record)
        except (ValueError, TypeError):
            continue
    return results


def _count_by_field(
    records: list[dict[str, Any]], field: str
) -> dict[str, int]:
    """Count records grouped by a specific field value.

    Args:
        records: List of record dicts.
        field: The field name to group by.

    Returns:
        Dict mapping field values to their counts.
    """
    counter: Counter[str] = Counter()
    for r in records:
        value = r.get(field, "unknown") or "unknown"
        counter[str(value)] += 1
    return dict(counter.most_common())


def _calculate_rate(
    numerator: int, denominator: int
) -> str:
    """Calculate a percentage rate as a formatted string.

    Args:
        numerator: The count of the target events.
        denominator: The total count.

    Returns:
        Formatted percentage string like ``"42.5%"``, or ``"N/A"`` if
        denominator is zero.
    """
    if denominator == 0:
        return "N/A"
    rate = (numerator / denominator) * 100
    return f"{rate:.1f}%"


def run_outreach_reporter(state: SEOAgentState) -> dict[str, Any]:
    """Generate a weekly markdown report of outreach activity.

    Queries Supabase for prospects discovered, scored, and queued this
    week, plus email send/open/reply metrics, sequence outcomes, link
    acquisitions, and template performance.

    Args:
        state: The current SEO agent state.

    Returns:
        State update with `report`, `errors`, and `next_node`.
    """
    errors: list[str] = list(state.get("errors", []))
    target_site = state["target_site"]
    week_start, _ = _get_week_boundaries()

    # -----------------------------------------------------------------------
    # Gather data
    # -----------------------------------------------------------------------

    # Prospects discovered this week
    new_prospects = _query_week_records("seo_backlink_prospects")
    prospects_by_method = _count_by_field(new_prospects, "discovery_method")

    # Prospects scored and queued
    scored_prospects = [
        p for p in new_prospects if p.get("status") in ("scored", "rejected")
    ]
    scored_by_tier = _count_by_field(
        [p for p in scored_prospects if p.get("tier")], "tier"
    )
    rejected_count = sum(
        1 for p in scored_prospects if p.get("status") == "rejected"
    )

    # Emails sent this week
    sent_emails = _query_week_records(
        "seo_outreach_emails",
        date_column="sent_at",
        extra_filters={"status": "sent"},
    )
    emails_by_tier = _count_by_field(sent_emails, "tier")
    emails_by_site = _count_by_field(sent_emails, "template_type")

    # Open and reply tracking
    total_sent = len(sent_emails)
    opened_count = sum(1 for e in sent_emails if e.get("opened", False))
    replied_count = sum(1 for e in sent_emails if e.get("replied", False))

    open_rate = _calculate_rate(opened_count, total_sent)
    reply_rate = _calculate_rate(replied_count, total_sent)

    # Sequences completed vs exhausted
    try:
        all_emails_this_week = _query_week_records("seo_outreach_emails")
    except Exception:
        all_emails_this_week = []

    exhausted_count = sum(
        1 for e in all_emails_this_week if e.get("status") == "exhausted"
    )
    completed_count = sum(
        1 for e in all_emails_this_week if e.get("status") == "replied"
    )

    # Links acquired (prospects with link confirmed)
    try:
        all_prospects = supabase_tools.query_table(
            "seo_backlink_prospects",
            filters={"status": "link_acquired"},
            limit=1000,
        )
    except Exception:
        all_prospects = []

    # Filter to current week
    links_acquired: list[dict[str, Any]] = []
    for p in all_prospects:
        created = p.get("created_at", "")
        if created and str(created) >= week_start:
            links_acquired.append(p)

    total_dr_value = sum(p.get("dr", 0) or 0 for p in links_acquired)
    avg_dr = (
        round(total_dr_value / len(links_acquired))
        if links_acquired
        else 0
    )

    # Top-performing template
    template_replies: Counter[str] = Counter()
    template_sends: Counter[str] = Counter()
    for e in sent_emails:
        tpl = e.get("template_type", "unknown") or "unknown"
        template_sends[tpl] += 1
        if e.get("replied", False):
            template_replies[tpl] += 1

    best_template = "N/A"
    best_template_rate = 0.0
    for tpl, sends in template_sends.items():
        if sends < 2:
            continue
        rate = template_replies.get(tpl, 0) / sends
        if rate > best_template_rate:
            best_template_rate = rate
            best_template = tpl

    # -----------------------------------------------------------------------
    # Build markdown report
    # -----------------------------------------------------------------------

    sections: list[str] = []

    sections.append(f"# Weekly Outreach Report — {target_site}")
    sections.append(f"**Week starting:** {week_start[:10]}")
    sections.append("")

    # Prospects discovered
    sections.append("## Prospects Discovered This Week")
    sections.append(f"**Total:** {len(new_prospects)}")
    if prospects_by_method:
        for method, count in prospects_by_method.items():
            sections.append(f"- {method}: {count}")
    else:
        sections.append("- No new prospects discovered")
    sections.append("")

    # Prospects scored
    sections.append("## Prospects Scored and Queued")
    sections.append(f"**Scored:** {len(scored_prospects) - rejected_count}")
    sections.append(f"**Rejected:** {rejected_count}")
    if scored_by_tier:
        for tier, count in scored_by_tier.items():
            sections.append(f"- {tier}: {count}")
    sections.append("")

    # Emails sent
    sections.append("## Emails Sent")
    sections.append(f"**Total sent:** {total_sent}")
    if emails_by_tier:
        sections.append("**By tier:**")
        for tier, count in emails_by_tier.items():
            sections.append(f"- Tier {tier}: {count}")
    if emails_by_site:
        sections.append("**By template:**")
        for tpl, count in emails_by_site.items():
            sections.append(f"- {tpl}: {count}")
    sections.append("")

    # Engagement
    sections.append("## Engagement Metrics")
    sections.append(f"- **Open rate:** {open_rate}")
    sections.append(f"- **Reply rate:** {reply_rate}")
    sections.append(f"- **Opened:** {opened_count}/{total_sent}")
    sections.append(f"- **Replied:** {replied_count}/{total_sent}")
    sections.append("")

    # Sequences
    sections.append("## Sequence Outcomes")
    sections.append(f"- **Sequences with replies:** {completed_count}")
    sections.append(f"- **Sequences exhausted:** {exhausted_count}")
    sections.append("")

    # Links acquired
    sections.append("## Links Acquired")
    sections.append(f"**Total:** {len(links_acquired)}")
    sections.append(f"**Estimated avg DR:** {avg_dr}")
    sections.append(f"**Total DR value:** {total_dr_value}")
    if links_acquired:
        for link in links_acquired[:10]:
            sections.append(
                f"- {link.get('domain', 'unknown')} (DR {link.get('dr', '?')})"
            )
    sections.append("")

    # Top template
    sections.append("## Top-Performing Email Template")
    if best_template != "N/A":
        sections.append(
            f"**{best_template}** — "
            f"{best_template_rate * 100:.1f}% reply rate "
            f"({template_replies.get(best_template, 0)} replies / "
            f"{template_sends.get(best_template, 0)} sends)"
        )
    else:
        sections.append("Not enough data to determine top template.")
    sections.append("")

    report = "\n".join(sections)

    logger.info(
        "Outreach report generated for %s (%d chars)",
        target_site,
        len(report),
    )

    return {
        "report": report,
        "errors": errors,
        "next_node": "END",
    }
