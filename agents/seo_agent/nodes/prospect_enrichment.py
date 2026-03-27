"""Prospect enrichment node — adds context and contact data to raw prospects.

For each new prospect, summarises the page, extracts contact patterns,
pulls DR/traffic from Ahrefs data, and checks competitor link overlap.
"""

from __future__ import annotations

import logging
from typing import Any

from agents.seo_agent.state import SEOAgentState
from agents.seo_agent.tools import llm_router, supabase_tools

logger = logging.getLogger(__name__)


def _summarise_page(
    page_url: str,
    page_title: str,
    *,
    weekly_spend: float,
    site: str,
) -> dict[str, Any]:
    """Use an LLM to generate a short summary of a prospect page.

    Args:
        page_url: The URL of the page to summarise.
        page_title: The title of the page, if known.
        weekly_spend: Current weekly LLM spend in USD.
        site: Target site name for cost logging.

    Returns:
        The LLM response dict including `text` and `cost_usd`.
    """
    return llm_router.call_llm(
        task="summarise_page",
        messages=[
            {
                "role": "user",
                "content": (
                    f"Summarise this web page in 2-3 sentences for a link-building "
                    f"outreach specialist. Focus on the topic, audience, and whether "
                    f"it links out to external resources.\n\n"
                    f"URL: {page_url}\n"
                    f"Title: {page_title}"
                ),
            },
        ],
        system="You are an SEO analyst. Be concise and factual.",
        weekly_spend=weekly_spend,
        site=site,
        log_fn=supabase_tools.log_llm_cost,
    )


def _extract_contact(
    domain: str,
    page_url: str,
    *,
    weekly_spend: float,
    site: str,
) -> dict[str, Any]:
    """Use an LLM to guess contact email patterns for a domain.

    Args:
        domain: The prospect domain.
        page_url: The prospect page URL.
        weekly_spend: Current weekly LLM spend in USD.
        site: Target site name for cost logging.

    Returns:
        The LLM response dict including `text` and `cost_usd`.
    """
    return llm_router.call_llm(
        task="extract_contact_email",
        messages=[
            {
                "role": "user",
                "content": (
                    f"Given this domain and page URL, suggest the most likely "
                    f"contact email pattern (e.g. editor@, hello@, firstname@). "
                    f"If you can identify the author name, include it.\n\n"
                    f"Domain: {domain}\n"
                    f"Page URL: {page_url}\n\n"
                    f"Respond in this exact format:\n"
                    f"email_pattern: <pattern>\n"
                    f"author_name: <name or unknown>\n"
                    f"confidence: <high/medium/low>"
                ),
            },
        ],
        system="You are an email research specialist. Be concise.",
        weekly_spend=weekly_spend,
        site=site,
        log_fn=supabase_tools.log_llm_cost,
    )


def _parse_contact_response(text: str) -> dict[str, str]:
    """Parse the structured LLM response for contact extraction.

    Args:
        text: The raw LLM response text.

    Returns:
        Dict with `email_pattern`, `author_name`, and `confidence`.
    """
    result: dict[str, str] = {
        "email_pattern": "",
        "author_name": "",
        "confidence": "low",
    }
    for line in text.strip().splitlines():
        line = line.strip()
        if line.lower().startswith("email_pattern:"):
            result["email_pattern"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("author_name:"):
            result["author_name"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("confidence:"):
            result["confidence"] = line.split(":", 1)[1].strip().lower()
    return result


def _get_prospects_from_state_or_db(
    state: SEOAgentState,
) -> list[dict[str, Any]]:
    """Get prospects to enrich from state or fall back to Supabase query.

    Args:
        state: The current SEO agent state.

    Returns:
        List of prospect dicts with status ``new``.
    """
    prospects = state.get("backlink_prospects", [])
    if prospects:
        return [p for p in prospects if p.get("status") == "new"]

    return supabase_tools.query_table(
        "seo_backlink_prospects",
        filters={"status": "new"},
        limit=200,
        order_by="created_at",
        order_desc=False,
    )


def run_prospect_enrichment(state: SEOAgentState) -> dict[str, Any]:
    """Enrich backlink prospects with page summaries, contacts, and metrics.

    Takes prospects from state or queries Supabase for prospects with status
    ``new``. For each prospect, summarises the page via LLM, extracts
    contact email patterns, and augments with DR/traffic data. Updates
    records in Supabase and sets status to ``enriched``.

    Args:
        state: The current SEO agent state.

    Returns:
        State update with `enriched_prospects`, `errors`, and `next_node`.
    """
    errors: list[str] = list(state.get("errors", []))
    target_site = state["target_site"]
    weekly_spend = state.get("llm_spend_this_week", 0.0)
    enriched: list[dict[str, Any]] = []

    prospects = _get_prospects_from_state_or_db(state)
    if not prospects:
        logger.info("No new prospects to enrich for %s", target_site)
        return {
            "enriched_prospects": [],
            "errors": errors,
            "next_node": "END",
        }

    logger.info("Enriching %d prospects for %s", len(prospects), target_site)

    for prospect in prospects:
        prospect_id = prospect.get("id", "")
        domain = prospect.get("domain", "")
        page_url = prospect.get("page_url", "")
        page_title = prospect.get("page_title", "")

        enrichment: dict[str, Any] = {}

        # Summarise the page
        try:
            summary_resp = _summarise_page(
                page_url,
                page_title,
                weekly_spend=weekly_spend,
                site=target_site,
            )
            enrichment["page_summary"] = summary_resp["text"]
            weekly_spend += summary_resp.get("cost_usd", 0.0)
        except Exception:
            msg = f"Failed to summarise page {page_url}"
            logger.warning(msg, exc_info=True)
            errors.append(msg)
            enrichment["page_summary"] = ""

        # Extract contact email pattern
        try:
            contact_resp = _extract_contact(
                domain,
                page_url,
                weekly_spend=weekly_spend,
                site=target_site,
            )
            contact_data = _parse_contact_response(contact_resp["text"])
            enrichment["contact_email"] = contact_data["email_pattern"]
            enrichment["author_name"] = contact_data["author_name"]
            weekly_spend += contact_resp.get("cost_usd", 0.0)
        except Exception:
            msg = f"Failed to extract contact for {domain}"
            logger.warning(msg, exc_info=True)
            errors.append(msg)
            enrichment["contact_email"] = ""
            enrichment["author_name"] = ""

        # Fetch DR from Ahrefs if not already known
        existing_dr = prospect.get("dr", 0)
        if not existing_dr or existing_dr == 0:
            try:
                from agents.seo_agent.tools.ahrefs_tools import get_domain_rating
                dr_data = get_domain_rating(prospect.get("domain", ""))
                if isinstance(dr_data, dict):
                    enrichment["dr"] = dr_data.get("domain_rating", 0)
                elif isinstance(dr_data, (int, float)):
                    enrichment["dr"] = dr_data
                else:
                    enrichment["dr"] = 0
                logger.info("Fetched DR for %s: %s", prospect.get("domain"), enrichment["dr"])
            except Exception:
                logger.warning("Failed to fetch DR for %s", prospect.get("domain"), exc_info=True)
                enrichment["dr"] = 0
        else:
            enrichment["dr"] = existing_dr
        enrichment["monthly_traffic"] = prospect.get("monthly_traffic", 0)

        # Check if the prospect links to competitors (preserve existing data)
        enrichment["links_to_competitor"] = prospect.get(
            "links_to_competitor", False
        )
        enrichment["competitor_names"] = prospect.get("competitor_names", [])

        # Update the record in Supabase
        update_data = {
            **enrichment,
            "status": "enriched",
        }

        if prospect_id:
            update_data["id"] = prospect_id

        try:
            if prospect_id:
                updated = supabase_tools.upsert_record(
                    "seo_backlink_prospects", update_data
                )
            else:
                updated = {**prospect, **update_data}
            enriched.append(updated)
        except Exception:
            msg = f"Failed to update enrichment data for prospect {domain}"
            logger.warning(msg, exc_info=True)
            errors.append(msg)
            enriched.append({**prospect, **enrichment, "status": "enriched"})

    logger.info(
        "Enrichment complete for %s: %d prospects enriched",
        target_site,
        len(enriched),
    )

    return {
        "enriched_prospects": enriched,
        "errors": errors,
        "next_node": "END",
    }
