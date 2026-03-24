"""Reporting node — generates a weekly SEO performance report.

Aggregates data from Supabase across all tracked metrics, uses the LLM
to produce a narrative summary, writes the report to file, and optionally
sends it via email using Resend.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

from agents.seo_agent.config import SITE_PROFILES
from agents.seo_agent.state import SEOAgentState
from agents.seo_agent.tools import file_tools, supabase_tools
from agents.seo_agent.tools.llm_router import call_llm

logger = logging.getLogger(__name__)


def _aggregate_data(target_site: str) -> dict[str, Any]:
    """Pull and aggregate key SEO metrics from Supabase.

    Args:
        target_site: The target site identifier, or empty string for all sites.

    Returns:
        Dict of aggregated metrics for the report.
    """
    filters: dict[str, str] | None = (
        {"target_site": target_site} if target_site else None
    )

    # Keyword opportunities
    keyword_opps = supabase_tools.query_table(
        "seo_keyword_opportunities",
        filters=filters,
        limit=500,
    )

    # Content gaps
    content_gaps = supabase_tools.query_table(
        "seo_content_gaps",
        filters=filters,
        limit=500,
    )

    # Content briefs
    briefs = supabase_tools.query_table(
        "seo_content_briefs",
        filters=filters,
        limit=100,
    )

    # Content drafts
    drafts = supabase_tools.query_table(
        "seo_content_drafts",
        filters=filters,
        limit=100,
    )

    # Rank history (most recent entries)
    rank_history = supabase_tools.query_table(
        "seo_rank_history",
        filters=filters,
        limit=200,
        order_by="created_at",
        order_desc=True,
    )

    # Backlink prospects
    backlink_prospects = supabase_tools.query_table(
        "seo_backlink_prospects",
        filters=filters,
        limit=200,
    )

    # Identify significant rank movements
    rank_movements: list[dict[str, Any]] = []
    for entry in rank_history:
        position = entry.get("position")
        previous = entry.get("previous_position")
        if position is not None and previous is not None:
            movement = previous - position
            if abs(movement) >= 3:
                rank_movements.append({
                    "keyword": entry.get("keyword", ""),
                    "position": position,
                    "previous_position": previous,
                    "movement": movement,
                    "direction": "improved" if movement > 0 else "declined",
                    "url": entry.get("url", ""),
                })

    # Sort movements by absolute change (biggest first)
    rank_movements.sort(key=lambda x: abs(x.get("movement", 0)), reverse=True)

    return {
        "keyword_opportunities_count": len(keyword_opps),
        "content_gaps_count": len(content_gaps),
        "briefs_created": len(briefs),
        "drafts_written": len(drafts),
        "drafts_published": len(
            [d for d in drafts if d.get("status") == "published"]
        ),
        "rank_entries": len(rank_history),
        "rank_movements": rank_movements[:20],
        "backlink_prospects_total": len(backlink_prospects),
        "backlink_prospects_contacted": len(
            [p for p in backlink_prospects if p.get("status") == "contacted"]
        ),
        "backlink_prospects_replied": len(
            [p for p in backlink_prospects if p.get("reply_received")]
        ),
        "content_gaps_by_funnel": _count_by_field(content_gaps, "funnel_stage"),
        "drafts_by_status": _count_by_field(drafts, "status"),
    }


def _count_by_field(
    records: list[dict[str, Any]], field: str
) -> dict[str, int]:
    """Count records grouped by a field value.

    Args:
        records: List of record dicts.
        field: The field name to group by.

    Returns:
        Dict mapping field values to counts.
    """
    counts: dict[str, int] = {}
    for record in records:
        value = record.get(field, "unknown")
        counts[value] = counts.get(value, 0) + 1
    return counts


def _build_report_data_summary(data: dict[str, Any]) -> str:
    """Build a text summary of aggregated data for the LLM prompt.

    Args:
        data: The aggregated data dict.

    Returns:
        A formatted string summarising the data.
    """
    movements_text = ""
    for m in data.get("rank_movements", [])[:10]:
        movements_text += (
            f"  - '{m['keyword']}': position {m['previous_position']} -> "
            f"{m['position']} ({'+' if m['movement'] > 0 else ''}"
            f"{m['movement']})\n"
        )

    gaps_text = ", ".join(
        f"{k}: {v}" for k, v in data.get("content_gaps_by_funnel", {}).items()
    )

    drafts_text = ", ".join(
        f"{k}: {v}" for k, v in data.get("drafts_by_status", {}).items()
    )

    return (
        f"Keyword opportunities discovered: {data['keyword_opportunities_count']}\n"
        f"Content gaps identified: {data['content_gaps_count']}\n"
        f"  By funnel stage: {gaps_text or 'none'}\n"
        f"Briefs created: {data['briefs_created']}\n"
        f"Drafts written: {data['drafts_written']}\n"
        f"  By status: {drafts_text or 'none'}\n"
        f"Content published: {data['drafts_published']}\n"
        f"Rank entries tracked: {data['rank_entries']}\n"
        f"Significant rank movements:\n{movements_text or '  None'}\n"
        f"Backlink prospects: {data['backlink_prospects_total']}\n"
        f"  Contacted: {data['backlink_prospects_contacted']}\n"
        f"  Replied: {data['backlink_prospects_replied']}\n"
    )


def _send_report_email(report_markdown: str, report_date: str) -> bool:
    """Send the report via Resend if configured.

    Args:
        report_markdown: The full report markdown content.
        report_date: The report date string for the subject line.

    Returns:
        True if the email was sent successfully, False otherwise.
    """
    api_key = os.getenv("RESEND_API_KEY", "")
    recipient = os.getenv("REPORT_EMAIL", "")

    if not api_key or not recipient:
        logger.info("Resend not configured — skipping email send")
        return False

    try:
        import resend

        resend.api_key = api_key
        resend.Emails.send({
            "from": "SEO Agent <seo@reports.noreply.com>",
            "to": [recipient],
            "subject": f"SEO Weekly Report — {report_date}",
            "text": report_markdown,
        })
        logger.info("Report email sent to %s", recipient)
        return True
    except ImportError:
        logger.warning("resend package not installed — skipping email send")
        return False
    except Exception:
        logger.warning("Failed to send report email", exc_info=True)
        return False


def run_reporting(state: SEOAgentState) -> dict[str, Any]:
    """Generate and distribute a weekly SEO performance report.

    Aggregates metrics from Supabase, uses the LLM to generate a narrative
    summary, writes the report to a markdown file, and optionally sends it
    via email using Resend.

    Args:
        state: The current SEO agent state.

    Returns:
        State update with `report`, `errors`, and `next_node`.
    """
    errors: list[str] = list(state.get("errors", []))
    target_site = state.get("target_site", "")
    report_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    # Step 1: Aggregate data from Supabase
    logger.info("Aggregating report data for '%s'", target_site or "all sites")
    try:
        data = _aggregate_data(target_site)
    except Exception:
        msg = "Failed to aggregate report data from Supabase"
        logger.error(msg, exc_info=True)
        errors.append(msg)
        return {
            "report": None,
            "errors": errors,
            "next_node": "END",
        }

    # Step 2: Generate narrative summary via LLM
    data_summary = _build_report_data_summary(data)

    messages = [
        {
            "role": "user",
            "content": (
                "Generate a concise weekly SEO report in markdown format "
                "based on the following data. Use these exact sections:\n\n"
                "1. **Top Ranking Movements** — highlight the biggest wins "
                "and losses\n"
                "2. **Content Published** — summarise content output\n"
                "3. **Backlink Pipeline** — outreach progress and replies\n"
                "4. **Content Gaps** — remaining opportunities by funnel stage\n"
                "5. **Priorities for Next 7 Days** — actionable recommendations\n\n"
                f"Report date: {report_date}\n"
                f"Target site: {target_site or 'all sites'}\n\n"
                f"Data:\n```\n{data_summary}```\n\n"
                "Keep it brief and actionable. Use bullet points. "
                "No filler or preamble."
            ),
        }
    ]

    try:
        llm_result = call_llm(
            task="weekly_report",
            messages=messages,
            weekly_spend=state.get("llm_spend_this_week", 0.0),
            site=target_site,
            log_fn=supabase_tools.log_llm_cost,
        )
        narrative = llm_result.get("text", "")
    except Exception:
        msg = "LLM call failed for weekly report narrative"
        logger.warning(msg, exc_info=True)
        errors.append(msg)
        # Fall back to a data-only report
        narrative = (
            f"# SEO Weekly Report — {report_date}\n\n"
            f"*Narrative generation failed. Raw data below.*\n\n"
            f"```\n{data_summary}```\n"
        )

    # Build the full report
    report_header = (
        f"# SEO Weekly Report — {report_date}\n\n"
        f"**Site:** {target_site or 'All Sites'}\n\n"
    )
    full_report = report_header + narrative

    # Step 3: Write the report to file
    report_name = f"seo-report-{report_date}"
    try:
        file_path = file_tools.write_report(report_name, full_report)
        logger.info("Report written to %s", file_path)
    except Exception:
        msg = f"Failed to write report file '{report_name}'"
        logger.warning(msg, exc_info=True)
        errors.append(msg)

    # Step 4: Optionally send via email
    _send_report_email(full_report, report_date)

    logger.info("Weekly report generated for %s", report_date)

    return {
        "report": full_report,
        "errors": errors,
        "next_node": "END",
    }
