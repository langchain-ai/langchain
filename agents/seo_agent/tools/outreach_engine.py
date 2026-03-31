"""Outreach engine — email generation, Instantly campaign management, CRM writes.

Orchestrates the full outreach pipeline: research targets → generate
personalised emails → create Instantly campaigns → track replies and links.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Site profiles for outreach (differentiators, sender, rules)
# ---------------------------------------------------------------------------

SITE_PROFILES: dict[str, dict[str, Any]] = {
    "freeroomplanner": {
        "url": "https://freeroomplanner.com",
        "description": (
            "Free browser-based floor planner for homeowners. Draw walls, "
            "add furniture, export PNG plans."
        ),
        "niche": "Free room planning tools for UK homeowners",
        "target_audience": (
            "UK homeowners, interior designers, kitchen/bathroom fitters, "
            "contractors"
        ),
        "differentiators": [
            "No sign-up required",
            "UK room size standards",
            "30+ furniture items",
            "Embed option for businesses",
        ],
        "sender_name": "Ben",
        "sender_email": "ben@freeroomplanner.com",
        "competitors": [
            "floorplanner.com",
            "roomsketcher.com",
            "planner5d.com",
            "homestyler.com",
        ],
        "preferred_outreach": ["inclusion", "guestpost", "mention"],
        "avoid": ["Competitor tool sites", "pay-to-play directories"],
    },
    "kitchen_estimator": {
        "url": "https://kitchencostestimator.com",
        "description": (
            "Interactive kitchen renovation cost estimator for UK/US/Canada. "
            "Step-by-step wizard with real pricing data."
        ),
        "niche": "Kitchen renovation cost calculators",
        "target_audience": (
            "UK homeowners budgeting for kitchen renovations"
        ),
        "differentiators": [
            "68 cost items",
            "26 regional multipliers",
            "No sign-up",
            "Country-specific pricing",
        ],
        "sender_name": "Ben",
        "sender_email": "ben@kitchencostestimator.com",
        "competitors": [
            "checkatrade.com",
            "householdquotes.co.uk",
            "mybuilder.com",
        ],
        "preferred_outreach": ["inclusion", "mention"],
        "avoid": ["Pay-for-lead sites"],
    },
    "ralf_seo": {
        "url": "https://ralfseo.com",
        "description": "An autonomous AI agent's field journal on SEO",
        "niche": "AI agent building, SEO case studies, autonomous systems",
        "target_audience": (
            "SEO practitioners, AI agent builders, indie developers"
        ),
        "differentiators": [
            "Real data from real sites",
            "Written by an actual running agent",
            "Honest about failures",
        ],
        "sender_name": "Ben",
        "sender_email": "ben@ralfseo.com",
        "competitors": [],
        "preferred_outreach": ["guestpost", "mention"],
        "avoid": ["Generic tech blogs", "listicle mills"],
    },
}

# ---------------------------------------------------------------------------
# Email generation via LLM
# ---------------------------------------------------------------------------

_TEMPLATE_PROMPTS: dict[str, str] = {
    "inclusion": (
        "Write a short, direct outreach email asking the recipient to include "
        "our tool in their resource list or roundup article.\n"
        "Structure: one-line intro → what the site does in one sentence → "
        "3 bullet differentiators → single ask → sign-off.\n"
        "Tone: short, direct, low friction."
    ),
    "guestpost": (
        "Write a professional outreach email proposing a guest post "
        "collaboration.\n"
        "Structure: intro + site description → two article angle options → "
        "word count + format offer → link ask → offer to send draft.\n"
        "Tone: professional, collaborative."
    ),
    "mention": (
        "Write a very brief, friendly outreach email suggesting they mention "
        "our tool in their content.\n"
        "Structure: one-sentence site description → one-line context → "
        "reciprocal offer → single question ask.\n"
        "Tone: friendly, very brief."
    ),
}


def generate_email(
    profile: dict[str, Any],
    target: dict[str, Any],
) -> dict[str, str]:
    """Generate a personalised outreach email using the LLM.

    Args:
        profile: Site profile from SITE_PROFILES.
        target: Target dict with article_url, article_title, contact_name, outreach_type.

    Returns:
        Dict with ``subject`` and ``body`` keys.
    """
    from agents.seo_agent.tools.llm_router import call_llm

    outreach_type = target.get("outreach_type", "inclusion")
    template_guidance = _TEMPLATE_PROMPTS.get(outreach_type, _TEMPLATE_PROMPTS["inclusion"])

    differentiators = "\n".join(f"- {d}" for d in profile.get("differentiators", []))

    prompt = f"""You are writing an outreach email on behalf of {profile['sender_name']}
from {profile['url']}.

TARGET:
- Their site/article: {target.get('article_url', target.get('url', 'N/A'))}
- Article title: {target.get('article_title', 'N/A')}
- Contact name: {{{{firstName}}}}

OUR SITE:
- URL: {profile['url']}
- Description: {profile['description']}
- Key differentiators:
{differentiators}

TEMPLATE TYPE: {outreach_type}
{template_guidance}

RULES:
- Personalise: reference their specific article/page
- Use {{{{firstName}}}} token for the recipient's name (Instantly will replace it)
- Use our differentiators naturally
- Include a subject line
- Single clear ask per email
- NEVER mention SEO, backlinks, domain rating, or link building
- Keep under 150 words
- Sign off as {profile['sender_name']}

Return your response in this exact format:
SUBJECT: <subject line>
BODY:
<email body>"""

    result = call_llm(
        task="write_tier2_email",
        messages=[{"role": "user", "content": prompt}],
        system="You are an expert outreach email writer. Write natural, personalised emails.",
        site=target.get("site_id", ""),
    )

    text = result["text"].strip()

    # Parse subject and body
    subject = ""
    body = text
    if "SUBJECT:" in text:
        parts = text.split("BODY:", 1)
        subject_line = parts[0].replace("SUBJECT:", "").strip()
        subject = subject_line.split("\n")[0].strip()
        body = parts[1].strip() if len(parts) > 1 else text

    return {"subject": subject, "body": body}


def generate_followup_sequence(
    profile: dict[str, Any],
    target: dict[str, Any],
    initial_email: dict[str, str],
) -> list[dict[str, str]]:
    """Generate touch 2 and touch 3 follow-up emails.

    Args:
        profile: Site profile from SITE_PROFILES.
        target: Target dict.
        initial_email: The initial email dict with subject and body.

    Returns:
        List of two dicts, each with ``subject``, ``body``, and ``delay_days``.
    """
    from agents.seo_agent.tools.llm_router import call_llm

    prompt = f"""Write two follow-up emails for an outreach sequence.

CONTEXT:
- We emailed {{{{firstName}}}} about including {profile['url']} in their content.
- Their article: {target.get('article_url', 'N/A')}
- Our initial email subject was: {initial_email['subject']}

FOLLOW-UP 1 (send day 5):
- Acknowledge they're busy
- Reiterate the ask briefly
- Keep it to 2-3 sentences

FOLLOW-UP 2 (send day 12):
- Final touch, gracious
- Offer to drop it if not interested
- Keep it to 2-3 sentences

RULES:
- Use {{{{firstName}}}} token
- NEVER mention SEO, backlinks, or DR
- Sign off as {profile['sender_name']}

Return in this exact format:
FOLLOWUP1_SUBJECT: <subject>
FOLLOWUP1_BODY:
<body>

FOLLOWUP2_SUBJECT: <subject>
FOLLOWUP2_BODY:
<body>"""

    result = call_llm(
        task="write_followup_email",
        messages=[{"role": "user", "content": prompt}],
        system="You are an expert outreach email writer. Write natural follow-ups.",
        site=target.get("site_id", ""),
    )

    text = result["text"].strip()
    followups: list[dict[str, str]] = []

    # Parse follow-up 1
    if "FOLLOWUP1_SUBJECT:" in text:
        parts = text.split("FOLLOWUP2_SUBJECT:", 1)
        f1_text = parts[0]
        f1_parts = f1_text.split("FOLLOWUP1_BODY:", 1)
        f1_subject = f1_parts[0].replace("FOLLOWUP1_SUBJECT:", "").strip().split("\n")[0]
        f1_body = f1_parts[1].strip() if len(f1_parts) > 1 else ""
        followups.append({"subject": f1_subject, "body": f1_body, "delay_days": "5"})

        # Parse follow-up 2
        if len(parts) > 1:
            f2_text = parts[1]
            f2_parts = f2_text.split("FOLLOWUP2_BODY:", 1)
            f2_subject = f2_parts[0].strip().split("\n")[0]
            f2_body = f2_parts[1].strip() if len(f2_parts) > 1 else ""
            followups.append({"subject": f2_subject, "body": f2_body, "delay_days": "12"})

    return followups


# ---------------------------------------------------------------------------
# Campaign creation pipeline
# ---------------------------------------------------------------------------


def create_outreach_campaign(
    profile: dict[str, Any],
    targets: list[dict[str, Any]],
    outreach_type: str,
    launch: bool = False,
    daily_limit: int = 30,
) -> dict[str, Any]:
    """Full pipeline: create campaign → build sequence → add leads → write to Supabase.

    Args:
        profile: Site profile from SITE_PROFILES.
        targets: List of target dicts (must have contact_email).
        outreach_type: One of inclusion, guestpost, mention.
        launch: Whether to launch the campaign immediately.
        daily_limit: Max emails per day.

    Returns:
        Summary dict with campaign_id, emails_added, etc.
    """
    from agents.seo_agent.tools import instantly_client
    from agents.seo_agent.tools.supabase_tools import insert_record, update_record

    site_id = next(
        (k for k, v in SITE_PROFILES.items() if v["url"] == profile["url"]),
        "unknown",
    )

    # Filter targets with valid emails
    valid_targets = [t for t in targets if t.get("contact_email")]
    if not valid_targets:
        return {"error": "No targets with contact emails", "emails_added": 0}

    # 1. Create Instantly campaign
    campaign_name = (
        f"{site_id}_{outreach_type}_{datetime.now(tz=timezone.utc).strftime('%Y%m%d')}"
    )
    campaign = instantly_client.create_campaign(
        name=campaign_name,
        sender_email=profile["sender_email"],
    )
    campaign_id = campaign.get("id", "")
    logger.info("Created Instantly campaign: %s (%s)", campaign_name, campaign_id)

    # 2. Generate emails and build leads
    leads: list[dict] = []
    for target in valid_targets:
        try:
            email = generate_email(profile, target)
            followups = generate_followup_sequence(profile, target, email)

            lead: dict[str, Any] = {
                "email": target["contact_email"],
                "first_name": target.get("contact_name", "").split()[0]
                if target.get("contact_name")
                else "",
                "company_name": target.get("site_name", ""),
                "custom_variables": {
                    "article_url": target.get("article_url", ""),
                    "article_title": target.get("article_title", ""),
                    "site_url": profile["url"],
                },
            }
            leads.append(lead)

            # Update target status
            if target.get("id"):
                try:
                    update_record(
                        "outreach_targets",
                        target["id"],
                        {"status": "campaign_created"},
                    )
                except Exception:
                    logger.debug("Could not update target status", exc_info=True)
        except Exception:
            logger.warning(
                "Failed to generate email for %s", target.get("contact_email"),
                exc_info=True,
            )

    # 3. Add leads to Instantly
    add_result = {"uploaded": 0, "duplicates": 0}
    if leads:
        try:
            add_result = instantly_client.add_leads(campaign_id, leads)
        except Exception:
            logger.error("Failed to add leads to Instantly", exc_info=True)
            add_result = {"error": "lead upload failed"}

    # 4. Optionally launch
    if launch and campaign_id:
        try:
            instantly_client.launch_campaign(campaign_id)
        except Exception:
            logger.error("Failed to launch campaign", exc_info=True)

    # 5. Write to Supabase
    try:
        insert_record("outreach_emails", {
            "site_id": site_id,
            "instantly_campaign_id": campaign_id,
            "instantly_campaign_name": campaign_name,
            "outreach_type": outreach_type,
            "emails_added": len(leads),
            "duplicates_skipped": add_result.get("duplicates", 0),
            "launched": launch,
            "daily_limit": daily_limit,
        })
    except Exception:
        logger.error("Failed to write outreach_emails record", exc_info=True)

    return {
        "campaign_id": campaign_id,
        "campaign_name": campaign_name,
        "emails_added": len(leads),
        "duplicates_skipped": add_result.get("duplicates", 0),
        "launched": launch,
    }


# ---------------------------------------------------------------------------
# Target research
# ---------------------------------------------------------------------------


def research_targets(site_id: str) -> list[dict[str, Any]]:
    """Use web search to find outreach targets for a site.

    Args:
        site_id: One of freeroomplanner, kitchen_estimator, ralf_seo.

    Returns:
        List of target dicts written to outreach_targets table.
    """
    from agents.seo_agent.tools.web_search_tools import search
    from agents.seo_agent.tools.supabase_tools import upsert_record

    profile = SITE_PROFILES.get(site_id)
    if not profile:
        return []

    queries = [
        f"best {profile['niche']} blog",
        f"{profile['niche']} resource list",
        f"{profile['niche']} tools roundup 2026",
        f"guest post {profile['niche']}",
        f"{profile['niche']} tips article",
    ]

    # Filter out competitor domains
    avoid_domains = set(profile.get("competitors", []) + profile.get("avoid", []))

    targets: list[dict[str, Any]] = []
    for query in queries:
        try:
            results = search(query, max_results=8)
            for r in results:
                url = r.get("url", "")
                title = r.get("title", "")

                # Skip competitors and avoid list
                skip = False
                for avoid in avoid_domains:
                    if isinstance(avoid, str) and avoid.lower() in url.lower():
                        skip = True
                        break
                if skip:
                    continue

                # Skip our own sites
                if profile["url"].replace("https://", "") in url:
                    continue

                target_data = {
                    "site_id": site_id,
                    "site_name": profile.get("url", site_id),
                    "url": url.split("?")[0],  # Strip query params
                    "article_url": url,
                    "article_title": title,
                    "outreach_type": _classify_outreach_type(
                        title, url, profile.get("preferred_outreach", [])
                    ),
                    "status": "queued",
                    "notes": f"Found via: {query}",
                }

                try:
                    record = upsert_record(
                        "outreach_targets",
                        target_data,
                        on_conflict="url,site_id",
                    )
                    targets.append(record)
                except Exception:
                    logger.debug("Upsert failed for %s", url, exc_info=True)
        except Exception:
            logger.warning("Search failed for query: %s", query, exc_info=True)

    logger.info("Found %d outreach targets for %s", len(targets), site_id)
    return targets


def _classify_outreach_type(
    title: str, url: str, preferred: list[str]
) -> str:
    """Classify the best outreach type based on the target article."""
    title_lower = title.lower()
    url_lower = url.lower()

    if any(w in title_lower for w in ["tool", "resource", "list", "roundup", "best", "top"]):
        return "inclusion" if "inclusion" in preferred else preferred[0] if preferred else "mention"
    if any(w in title_lower for w in ["guest", "write for us", "contributor", "submit"]):
        return "guestpost" if "guestpost" in preferred else preferred[0] if preferred else "mention"
    if any(w in url_lower for w in ["blog", "article", "post", "guide", "how-to"]):
        return "mention" if "mention" in preferred else preferred[0] if preferred else "mention"

    return preferred[0] if preferred else "mention"


# ---------------------------------------------------------------------------
# Status and reporting
# ---------------------------------------------------------------------------


def get_outreach_status(site_id: str | None = None) -> dict[str, Any]:
    """Query outreach_targets grouped by status.

    Args:
        site_id: Filter by site, or None for all sites.

    Returns:
        Dict with status counts and site breakdown.
    """
    from agents.seo_agent.tools.supabase_tools import query_table

    filters = {"site_id": site_id} if site_id else None
    targets = query_table("outreach_targets", filters=filters, limit=500)

    by_status: dict[str, int] = {}
    by_site: dict[str, int] = {}
    for t in targets:
        status = t.get("status", "unknown")
        by_status[status] = by_status.get(status, 0) + 1
        sid = t.get("site_id", "unknown")
        by_site[sid] = by_site.get(sid, 0) + 1

    # Campaign stats
    emails = query_table("outreach_emails", filters=filters, limit=100)
    total_sent = sum(e.get("emails_added", 0) for e in emails)
    campaigns_launched = sum(1 for e in emails if e.get("launched"))

    # Replies
    reply_filters = {"site_id": site_id} if site_id else None
    replies = query_table("outreach_replies", filters=reply_filters, limit=100)

    # Confirmed links
    link_filters = {"site_id": site_id} if site_id else None
    links = query_table("outreach_links", filters=link_filters, limit=100)
    live_links = [l for l in links if l.get("is_live")]

    return {
        "total_targets": len(targets),
        "by_status": by_status,
        "by_site": by_site,
        "total_emails_sent": total_sent,
        "campaigns_launched": campaigns_launched,
        "total_campaigns": len(emails),
        "total_replies": len(replies),
        "total_links": len(links),
        "live_links": len(live_links),
    }


def get_outreach_report(site_id: str | None = None) -> str:
    """Generate a full text outreach report.

    Args:
        site_id: Filter by site, or None for all sites.

    Returns:
        Formatted report string.
    """
    status = get_outreach_status(site_id)

    lines = ["Outreach Report"]
    if site_id:
        lines[0] += f" — {site_id}"
    lines.append("")

    lines.append(f"Targets: {status['total_targets']}")
    for s, count in sorted(status["by_status"].items()):
        lines.append(f"  {s}: {count}")

    lines.append(f"\nCampaigns: {status['total_campaigns']} ({status['campaigns_launched']} launched)")
    lines.append(f"Emails sent: {status['total_emails_sent']}")
    lines.append(f"Replies: {status['total_replies']}")
    lines.append(f"Links confirmed: {status['total_links']} ({status['live_links']} live)")

    if status["by_site"]:
        lines.append("\nBy site:")
        for sid, count in sorted(status["by_site"].items()):
            lines.append(f"  {sid}: {count} targets")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Link confirmation
# ---------------------------------------------------------------------------


def confirm_live_link(
    site_id: str,
    target_url: str,
    link_url: str,
    anchor_text: str,
    dr: int | None = None,
    do_follow: bool = True,
) -> dict[str, Any]:
    """Record a confirmed backlink in the outreach_links table.

    Args:
        site_id: The site that received the link.
        target_url: The page that contains the link.
        link_url: The URL being linked to (our page).
        anchor_text: The anchor text used.
        dr: Domain rating of the linking site.
        do_follow: Whether the link is dofollow.

    Returns:
        The inserted record.
    """
    from agents.seo_agent.tools.supabase_tools import insert_record, query_table

    # Try to find matching target
    targets = query_table(
        "outreach_targets",
        filters={"site_id": site_id, "url": target_url},
        limit=1,
    )
    target_id = targets[0]["id"] if targets else None

    record = insert_record("outreach_links", {
        "site_id": site_id,
        "target_id": target_id,
        "target_url": target_url,
        "link_url": link_url,
        "anchor_text": anchor_text,
        "domain_rating": dr,
        "do_follow": do_follow,
        "is_live": True,
    })

    # Update target status if found
    if target_id:
        try:
            from agents.seo_agent.tools.supabase_tools import update_record
            update_record("outreach_targets", target_id, {"status": "link_confirmed"})
        except Exception:
            logger.debug("Could not update target status to link_confirmed", exc_info=True)

    logger.info(
        "Confirmed link: %s → %s (anchor: %s, DR: %s)",
        target_url, link_url, anchor_text, dr,
    )
    return record
