"""Email generator node — creates personalised outreach emails for scored prospects.

Selects a template based on discovery method, uses Opus for tier-1 and
Sonnet for tier-2 prospects, and saves generated emails to Supabase.
"""

from __future__ import annotations

import logging
from typing import Any

from agents.seo_agent.config import MIN_OUTREACH_SCORE, TIER1_OUTREACH_SCORE
from agents.seo_agent.state import SEOAgentState
from agents.seo_agent.tools import llm_router, supabase_tools

logger = logging.getLogger(__name__)

_UNSUBSCRIBE_LINE = (
    "\n\nIf you'd prefer not to hear from us, just reply with "
    "'unsubscribe' and I'll remove you immediately."
)

# Template names mapped from discovery method
_TEMPLATE_MAP: dict[str, str] = {
    "competitor_backlink": "resource_addition",
    "content_explorer": "resource_addition",
    "broken_link": "broken_link_replacement",
    "unlinked_mention": "unlinked_mention",
    "resource_page": "data_research_collaboration",
}
_DEFAULT_TEMPLATE = "expert_quote_offer"

# Template descriptions for LLM context
_TEMPLATE_DESCRIPTIONS: dict[str, str] = {
    "resource_addition": (
        "Template 1 — Resource Addition: You found their page while "
        "researching the topic and believe your resource would add genuine "
        "value to their readers."
    ),
    "broken_link_replacement": (
        "Template 2 — Broken Link Replacement: You noticed a broken link "
        "on their page and have a working replacement resource."
    ),
    "unlinked_mention": (
        "Template 3 — Unlinked Mention: They already mentioned the brand "
        "or site but did not include a hyperlink."
    ),
    "data_research_collaboration": (
        "Template 4 — Data/Research Collaboration: You have original data, "
        "research, or an interactive tool that could enhance their content."
    ),
    "expert_quote_offer": (
        "Template 5 — Expert Quote Offer: You are offering an expert quote "
        "or insight relevant to their audience's interests."
    ),
}


def _select_template(discovery_method: str) -> str:
    """Select an email template based on the prospect's discovery method.

    Args:
        discovery_method: How the prospect was discovered.

    Returns:
        Template key string.
    """
    return _TEMPLATE_MAP.get(discovery_method, _DEFAULT_TEMPLATE)


def _build_system_prompt(*, tier: str, template: str) -> str:
    """Build the system prompt for email generation.

    Args:
        tier: Prospect tier (``tier1`` or ``tier2``).
        template: The template key.

    Returns:
        System prompt string with writing rules and template context.
    """
    template_desc = _TEMPLATE_DESCRIPTIONS.get(template, "")

    # Import outreach strategy for segment-aware emails
    try:
        from agents.seo_agent.outreach_strategy import OUTREACH_SEGMENTS, OUTREACH_EMAIL_PROMPT
        strategy_context = (
            "\n\nOUTREACH PHILOSOPHY:\n"
            "- Lead with THEIR benefit, not ours. We are collaborators, not beggars.\n"
            "- Kitchen/bathroom providers: offer free room planner embed for their customers.\n"
            "- Bloggers: offer exclusive data, guest post exchange, tool features.\n"
            "- Influencers: offer free tools for their audience, challenges, cross-promo.\n"
            "- Resource pages: brief pitch, free tool, no catch.\n"
            "- Journalists: lead with a story angle and data.\n"
            "- Sign off as Ben (the owner), not Ralf (the agent).\n"
        )
    except ImportError:
        strategy_context = ""

    rules = (
        "Write a personalised outreach email following these rules strictly:\n"
        "- Address the recipient by their first name if known\n"
        "- Reference a specific page or piece of their content\n"
        "- Lead with what's in it for THEM — why linking to us benefits their audience\n"
        "- Be specific about what we offer in return (free tool, data, cross-promotion)\n"
        "- NEVER beg for links. Frame it as a collaboration between peers.\n"
        "- Keep the ask low-friction (one click, one reply)\n"
        "- Write the subject line like a human — short, specific, not salesy\n"
        "- Sign off as 'Ben' not 'Ralf'\n"
        + strategy_context
    )

    if tier == "tier1":
        rules += "- Maximum 150 words in the body\n"
        rules += "- This is a high-value prospect: invest in deep personalisation\n"
    else:
        rules += "- Maximum 120 words in the body\n"
        rules += "- Use personalisation tokens where specific data is available\n"
        rules += "- Keep it shorter and more direct than a tier-1 email\n"

    rules += f"\nTemplate type: {template_desc}\n"
    rules += (
        "\nRespond in this exact format:\n"
        "SUBJECT: <subject line>\n"
        "BODY:\n<email body>"
    )

    return rules


def _build_user_message(prospect: dict[str, Any]) -> str:
    """Build the user message containing prospect context for the LLM.

    Args:
        prospect: The scored prospect record.

    Returns:
        A formatted string with all available prospect data.
    """
    parts = [
        f"Domain: {prospect.get('domain', 'unknown')}",
        f"Page URL: {prospect.get('page_url', 'unknown')}",
        f"Page Title: {prospect.get('page_title', 'unknown')}",
        f"Page Summary: {prospect.get('page_summary', 'not available')}",
        f"Author Name: {prospect.get('author_name', 'unknown')}",
        f"Contact Email: {prospect.get('contact_email', 'unknown')}",
        f"DR: {prospect.get('dr', 'unknown')}",
        f"Discovery Method: {prospect.get('discovery_method', 'unknown')}",
        f"Links to Competitors: {prospect.get('links_to_competitor', False)}",
        f"Competitor Names: {', '.join(prospect.get('competitor_names', []))}",
    ]

    # Add broken link context if applicable
    if prospect.get("dead_url"):
        parts.append(f"Dead URL (broken link): {prospect['dead_url']}")
    if prospect.get("anchor"):
        parts.append(f"Broken Link Anchor Text: {prospect['anchor']}")

    return "\n".join(parts)


def _parse_email_response(text: str) -> dict[str, str]:
    """Parse the LLM response into subject and body components.

    Args:
        text: Raw LLM response text.

    Returns:
        Dict with `subject` and `body` keys.
    """
    subject = ""
    body = ""

    lines = text.strip().splitlines()
    in_body = False

    for line in lines:
        if line.strip().upper().startswith("SUBJECT:"):
            subject = line.split(":", 1)[1].strip()
        elif line.strip().upper().startswith("BODY:"):
            in_body = True
        elif in_body:
            body += line + "\n"

    return {
        "subject": subject,
        "body": body.strip(),
    }


def _get_scored_prospects(state: SEOAgentState) -> list[dict[str, Any]]:
    """Get scored prospects from state or fall back to Supabase query.

    Args:
        state: The current SEO agent state.

    Returns:
        List of scored prospect dicts meeting the minimum score threshold.
    """
    prospects = state.get("scored_prospects", [])
    if prospects:
        return [
            p
            for p in prospects
            if p.get("status") == "scored"
            and (p.get("score", 0) or 0) >= MIN_OUTREACH_SCORE
        ]

    return supabase_tools.query_table(
        "seo_backlink_prospects",
        filters={"status": "scored"},
        limit=200,
        order_by="score",
        order_desc=True,
    )


def run_email_generator(state: SEOAgentState) -> dict[str, Any]:
    """Generate personalised outreach emails for scored prospects.

    Selects a template based on each prospect's discovery method. Tier-1
    prospects (score 65+) use Opus for maximum quality; tier-2 (score 35-64)
    use Sonnet. All emails receive an unsubscribe line and are saved to
    Supabase with status ``queued``.

    Args:
        state: The current SEO agent state.

    Returns:
        State update with `emails_generated`, `errors`, and `next_node`.
    """
    errors: list[str] = list(state.get("errors", []))
    target_site = state["target_site"]
    weekly_spend = state.get("llm_spend_this_week", 0.0)
    emails_generated: list[dict[str, Any]] = []

    prospects = _get_scored_prospects(state)
    if not prospects:
        logger.info("No scored prospects to generate emails for %s", target_site)
        return {
            "emails_generated": [],
            "errors": errors,
            "next_node": "END",
        }

    logger.info(
        "Generating emails for %d scored prospects for %s",
        len(prospects),
        target_site,
    )

    for prospect in prospects:
        prospect_id = prospect.get("id", "")
        domain = prospect.get("domain", "")
        score = prospect.get("score", 0) or 0
        discovery_method = prospect.get("discovery_method", "")

        # Determine tier and LLM task
        if score >= TIER1_OUTREACH_SCORE:
            tier = "tier1"
            task = "write_tier1_email"
            tier_num = 1
        else:
            tier = "tier2"
            task = "write_tier2_email"
            tier_num = 2

        template = _select_template(discovery_method)
        system_prompt = _build_system_prompt(tier=tier, template=template)
        user_message = _build_user_message(prospect)

        try:
            llm_resp = llm_router.call_llm(
                task=task,
                messages=[{"role": "user", "content": user_message}],
                system=system_prompt,
                weekly_spend=weekly_spend,
                site=target_site,
                log_fn=supabase_tools.log_llm_cost,
            )
            weekly_spend += llm_resp.get("cost_usd", 0.0)
        except Exception:
            msg = f"LLM email generation failed for prospect {domain} ({task})"
            logger.warning(msg, exc_info=True)
            errors.append(msg)
            continue

        parsed = _parse_email_response(llm_resp["text"])
        subject = parsed["subject"]
        body = parsed["body"] + _UNSUBSCRIBE_LINE

        # Save to Supabase
        email_record: dict[str, Any] = {
            "prospect_id": prospect_id if prospect_id else None,
            "subject": subject,
            "body": body,
            "tier": tier_num,
            "template_type": template,
            "sequence_step": 0,
            "status": "queued",
        }

        try:
            saved = supabase_tools.insert_record(
                "seo_outreach_emails", email_record
            )
            emails_generated.append(saved)
            logger.debug(
                "Generated %s email for %s (score %d, template %s)",
                tier,
                domain,
                score,
                template,
            )
        except Exception:
            msg = f"Failed to save email for prospect {domain} to Supabase"
            logger.warning(msg, exc_info=True)
            errors.append(msg)
            emails_generated.append(email_record)

    logger.info(
        "Email generation complete for %s: %d emails created",
        target_site,
        len(emails_generated),
    )

    return {
        "emails_generated": emails_generated,
        "errors": errors,
        "next_node": "END",
    }
