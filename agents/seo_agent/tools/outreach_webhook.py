"""Outreach webhook/polling — checks for new replies and bounces from Instantly."""

import logging
from typing import Any

from agents.seo_agent.tools.instantly_client import list_replies, get_campaign_stats
from agents.seo_agent.tools.supabase_tools import insert_record, query_table, get_client

logger = logging.getLogger(__name__)


def poll_instantly_replies() -> dict[str, Any]:
    """Poll Instantly for new replies and update CRM.

    Returns summary of new replies found.
    """
    new_replies = 0
    updated_targets = 0
    campaign_ids: set[str] = set()

    try:
        # Get all campaign IDs we've created
        campaigns = query_table("outreach_emails", limit=100)
        campaign_ids = {c.get("instantly_campaign_id") for c in campaigns if c.get("instantly_campaign_id")}

        # Check existing reply emails to avoid duplicates
        existing_replies = query_table("outreach_replies", limit=1000)
        existing_from = {r.get("from_address", "").lower() for r in existing_replies}

        for campaign_id in campaign_ids:
            try:
                replies = list_replies(campaign_id=campaign_id, limit=50)

                for reply in replies:
                    from_addr = reply.get("from_address", "").lower()
                    if from_addr in existing_from:
                        continue

                    # Determine site_id from campaign
                    campaign_record = next(
                        (c for c in campaigns if c.get("instantly_campaign_id") == campaign_id),
                        {},
                    )
                    site_id = campaign_record.get("site_id", "")

                    # Insert reply
                    try:
                        insert_record("outreach_replies", {
                            "site_id": site_id,
                            "instantly_campaign_id": campaign_id,
                            "from_address": from_addr,
                            "subject": reply.get("subject", ""),
                            "body": reply.get("body", "")[:2000],
                            "sentiment": classify_reply_sentiment(reply.get("body", "")),
                        })
                        new_replies += 1
                        existing_from.add(from_addr)
                    except Exception:
                        logger.warning("Failed to save reply from %s", from_addr, exc_info=True)

                    # Update target status to "replied"
                    try:
                        targets = query_table(
                            "outreach_targets",
                            filters={"contact_email": from_addr},
                            limit=1,
                        )
                        if targets:
                            client = get_client()
                            client.table("outreach_targets").update(
                                {"status": "replied"}
                            ).eq("contact_email", from_addr).execute()
                            updated_targets += 1
                    except Exception:
                        logger.warning("Failed to update target status for %s", from_addr, exc_info=True)

            except Exception:
                logger.warning("Failed to poll replies for campaign %s", campaign_id, exc_info=True)

    except Exception as e:
        logger.error("Reply polling failed: %s", e, exc_info=True)

    summary = {
        "new_replies": new_replies,
        "targets_updated": updated_targets,
        "campaigns_checked": len(campaign_ids),
    }
    logger.info("Reply poll complete: %s", summary)
    return summary


def classify_reply_sentiment(body: str) -> str:
    """Use LLM to classify reply sentiment."""
    if not body or not body.strip():
        return "neutral"
    try:
        from agents.seo_agent.tools.llm_router import call_llm

        result = call_llm(
            task="classify_sentiment",
            messages=[{
                "role": "user",
                "content": (
                    "Classify this email reply sentiment as one of: "
                    "positive, not_interested, paid_only, neutral.\n\n"
                    f"Reply:\n{body[:500]}\n\n"
                    "Respond with ONLY the classification word."
                ),
            }],
            system="You classify email reply sentiment. Respond with exactly one word.",
        )
        text = result.get("text", "neutral").strip().lower()
        if text in ("positive", "not_interested", "paid_only", "neutral"):
            return text
        return "neutral"
    except Exception:
        return "neutral"
