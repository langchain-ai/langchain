"""Outreach strategy engine — intelligent, collaborative backlink acquisition.

Defines distinct outreach approaches for different target types, crafts
tailored pitches based on mutual value, and tracks progress against goals.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Target segments — each gets a different approach
# ---------------------------------------------------------------------------

OUTREACH_SEGMENTS: dict[str, dict[str, Any]] = {
    "kitchen_bathroom_providers": {
        "description": "Kitchen and bathroom companies, fitters, and showrooms",
        "sites": ["freeroomplanner", "kitchensdirectory"],
        "approach": "embed_partnership",
        "value_proposition": (
            "We offer a free room planner tool that your customers can use to design "
            "their kitchen or bathroom layout before visiting your showroom. Embedding a "
            "link to freeroomplanner.com on your website means your customers arrive with "
            "a clear plan — saving you consultation time and increasing conversion rates. "
            "It's completely free and requires no sign-up, so there's zero friction for "
            "your customers."
        ),
        "what_we_offer": [
            "Free room planner tool their customers can use (no sign-up, no cost)",
            "Co-branded landing page if volume justifies it",
            "Featured listing on kitchensdirectory.co.uk (verified makers)",
            "Cross-promotion to our audience of homeowners planning renovations",
        ],
        "what_we_ask": [
            "Add a 'Plan Your Kitchen' or 'Design Your Room' link to their website",
            "Link to freeroomplanner.com from their resources or tools page",
            "Optional: listing on kitchensdirectory.co.uk if they're a UK maker",
        ],
        "email_tone": "professional, partnership-focused, emphasise mutual benefit",
        "subject_templates": [
            "Free tool for your customers — room planner partnership",
            "Help your customers plan before they visit — free tool",
            "Partnership idea: free room planner for {company} customers",
        ],
        "discovery_queries": [
            "kitchen showroom UK website",
            "bathroom fitters website UK",
            "kitchen design company resources page",
            "fitted kitchen company website UK",
            "bathroom renovation company UK",
        ],
    },
    "home_interior_bloggers": {
        "description": "Home interiors, renovation, and design bloggers",
        "sites": ["freeroomplanner", "kitchensdirectory", "kitchen_estimator"],
        "approach": "content_collaboration",
        "value_proposition": (
            "We have tools and data that make your content more useful to your readers. "
            "Our free room planner lets readers actually try out the layouts you write about. "
            "Our kitchen cost estimator gives real pricing data for renovation articles. "
            "We'd love to collaborate on content that references these tools — your readers "
            "get practical value, and we both get exposure."
        ),
        "what_we_offer": [
            "Exclusive data/stats from our kitchen cost database for their articles",
            "Guest post exchange — we write for them, they write for us",
            "Feature their content in our guides section (cross-linking)",
            "Free use of our tools with attribution",
            "Interview/quote opportunities as 'renovation tech' source",
        ],
        "what_we_ask": [
            "Link to our tool in relevant articles (room planner, cost estimator)",
            "Include us in resource roundups (e.g. 'best free room planners')",
            "Guest post with natural link back",
            "Mention in social media when discussing room planning",
        ],
        "email_tone": "friendly, collaborative, creative, suggest specific content ideas",
        "subject_templates": [
            "Content collab idea — {specific_topic} for your readers",
            "Free tool + data for your renovation content",
            "Loved your post on {their_topic} — quick idea",
        ],
        "discovery_queries": [
            "home interior blog UK",
            "kitchen renovation blog",
            "home improvement blogger UK",
            "interior design blog resources",
            "renovation diary blog UK",
        ],
    },
    "home_improvement_influencers": {
        "description": "YouTube, Instagram, TikTok home renovation influencers",
        "sites": ["freeroomplanner", "kitchen_estimator"],
        "approach": "influencer_collaboration",
        "value_proposition": (
            "We've built free tools that your audience would genuinely find useful — "
            "a room planner for visualising layouts and a cost estimator for budgeting. "
            "Rather than a paid promotion, we'd love to explore a genuine collaboration "
            "where your audience gets something useful and we both grow."
        ),
        "what_we_offer": [
            "Free tools for their audience (no affiliate fees, genuinely free)",
            "Custom room planning challenge they can set for their followers",
            "Exclusive 'renovation cost data' for their content",
            "Cross-promotion on our sites (featured in our inspiration section)",
            "Potential for ongoing partnership as our tools grow",
        ],
        "what_we_ask": [
            "Demo/mention the room planner in a renovation video",
            "Link in bio or video description to our tool",
            "Include in a 'tools I use for renovation planning' post",
        ],
        "email_tone": "casual, enthusiastic, fan-first, suggest specific collaboration format",
        "subject_templates": [
            "Free room planner for your followers — collab idea",
            "Loved your {their_content} — tool your audience would like",
            "Quick idea: room planning challenge for your audience?",
        ],
        "discovery_queries": [
            "home renovation youtube UK",
            "kitchen renovation influencer",
            "interior design instagram UK",
            "home improvement tiktok UK",
            "DIY renovation content creator UK",
        ],
    },
    "resource_page_targets": {
        "description": "Sites with resource/links/tools pages in home improvement niche",
        "sites": ["freeroomplanner", "kitchen_estimator"],
        "approach": "resource_inclusion",
        "value_proposition": (
            "We noticed you have a resources page for your readers. We've built a completely "
            "free room planner (no sign-up needed) that would be a useful addition. It's "
            "already used by homeowners across the UK for planning kitchens, bathrooms, "
            "and living rooms."
        ),
        "what_we_offer": [
            "A genuinely useful free tool for their audience",
            "No affiliate complexity — it's free forever",
            "We'll link back to their site from our resources section",
        ],
        "what_we_ask": [
            "Add freeroomplanner.com to their resources/tools page",
        ],
        "email_tone": "brief, respectful, no-pressure, focused on reader value",
        "subject_templates": [
            "Free room planner for your resources page",
            "Useful tool for your readers — free, no sign-up",
        ],
        "discovery_queries": [
            "intitle:resources home renovation UK",
            "intitle:useful links interior design",
            "intitle:recommended tools kitchen planning",
            "inurl:resources home improvement",
        ],
    },
    "pr_journalists": {
        "description": "Property, home, and lifestyle journalists",
        "sites": ["freeroomplanner", "kitchensdirectory", "kitchen_estimator"],
        "approach": "pr_story",
        "value_proposition": (
            "We have original data and tools that make good stories: how much kitchens "
            "really cost in different UK regions, a free room planner used by thousands "
            "of homeowners, and a curated directory of Britain's handmade kitchen makers. "
            "We're available for expert comment on home renovation trends."
        ),
        "what_we_offer": [
            "Original data: kitchen costs by region, popular styles, budget breakdowns",
            "Expert comment on renovation trends and costs",
            "Story angle: 'handmade kitchens cheaper than high-street' (our directory data)",
            "Story angle: 'the tools homeowners actually use to plan renovations'",
            "Story angle: 'how much does a kitchen really cost in 2026?'",
        ],
        "what_we_ask": [
            "Coverage with link to relevant site",
            "Inclusion in resource lists or roundups",
            "Expert source attribution with link",
        ],
        "email_tone": "professional, concise, lead with the story angle, respect their time",
        "subject_templates": [
            "Data: what kitchens really cost in {region} (2026)",
            "Story idea: handmade kitchens cheaper than you think",
            "Expert source: home renovation costs and planning tools",
        ],
        "discovery_queries": [
            "property journalist UK",
            "home renovation journalist",
            "homes and interiors writer UK",
            "HARO home improvement",
        ],
    },
    "interior_designers": {
        "description": "Interior designers — freelance, studios, and firms",
        "sites": ["freeroomplanner", "kitchensdirectory"],
        "approach": "tool_partnership",
        "value_proposition": (
            "Our free room planner lets your clients sketch layouts before your first "
            "consultation — they arrive with clear ideas, you save time on initial scoping. "
            "No cost, no sign-up, no branding requirements. Some designers embed the link "
            "on their website as a 'plan your room' tool for prospective clients."
        ),
        "what_we_offer": [
            "Free room planner tool their clients can use before consultations",
            "Featured listing on kitchensdirectory.co.uk designer section (if applicable)",
            "Cross-promotion to homeowners actively planning renovations",
            "Optional co-branded landing page for high-volume partners",
        ],
        "what_we_ask": [
            "Add a 'Plan Your Room' link on their website or client resources page",
            "Recommend freeroomplanner.com to clients during the planning phase",
            "Optional: backlink from their portfolio or resources page",
        ],
        "email_tone": "professional, design-aware, respect their creative expertise",
        "subject_templates": [
            "Free tool for your clients — room layout planner",
            "Help clients arrive prepared — free room planner",
            "Partnership idea: free planning tool for {company} clients",
        ],
        "discovery_queries": [
            "interior designer UK website",
            "interior design studio London",
            "residential interior designer portfolio",
            "kitchen interior designer UK",
            "bathroom interior designer UK",
            "interior design firm resources",
        ],
    },
}


# ---------------------------------------------------------------------------
# Email templates — tailored per segment
# ---------------------------------------------------------------------------


def generate_outreach_email(
    segment: str,
    prospect: dict,
    site: str = "freeroomplanner",
) -> dict[str, str]:
    """Generate a tailored outreach email for a prospect.

    Returns a dict with subject, body, and segment info. The actual email
    generation uses Claude via the LLM router for personalisation.
    """
    seg = OUTREACH_SEGMENTS.get(segment)
    if not seg:
        seg = OUTREACH_SEGMENTS["resource_page_targets"]  # default

    template_context = {
        "segment": segment,
        "approach": seg["approach"],
        "tone": seg["email_tone"],
        "value_proposition": seg["value_proposition"],
        "what_we_offer": seg["what_we_offer"],
        "what_we_ask": seg["what_we_ask"],
        "subject_templates": seg["subject_templates"],
        "prospect_domain": prospect.get("domain", ""),
        "prospect_name": prospect.get("author_name", ""),
        "prospect_page_title": prospect.get("page_title", ""),
        "prospect_page_summary": prospect.get("page_summary", ""),
        "prospect_dr": prospect.get("dr", 0),
        "target_site": site,
    }

    return template_context


# ---------------------------------------------------------------------------
# Outreach goals and progress tracking
# ---------------------------------------------------------------------------

MONTHLY_OUTREACH_GOALS: dict[str, int] = {
    "kitchen_bathroom_providers": 20,  # Partner embeds
    "home_interior_bloggers": 15,      # Content collaborations
    "home_improvement_influencers": 5,  # Influencer partnerships
    "resource_page_targets": 10,       # Resource page inclusions
    "pr_journalists": 5,              # PR/media outreach
    "interior_designers": 15,          # Designer partnerships
}


def get_outreach_progress(communications: list[dict]) -> dict[str, dict[str, int]]:
    """Calculate outreach progress against monthly goals.

    Returns progress per segment with sent, replied, and link_acquired counts.
    """
    progress: dict[str, dict[str, int]] = {}
    for segment, goal in MONTHLY_OUTREACH_GOALS.items():
        # Count comms that mention this segment in notes
        relevant = [c for c in communications if segment in (c.get("notes", "") or "")]
        progress[segment] = {
            "goal": goal,
            "sent": len([c for c in relevant if c.get("status") in ("sent", "opened", "replied")]),
            "replied": len([c for c in relevant if c.get("status") == "replied"]),
            "link_acquired": 0,  # Would need cross-reference with prospects
        }
    return progress


# ---------------------------------------------------------------------------
# Prompt for LLM email generation — segment-aware
# ---------------------------------------------------------------------------

OUTREACH_EMAIL_PROMPT = """Write a short, personalised outreach email for a backlink collaboration.

SEGMENT: {segment}
APPROACH: {approach}
TONE: {tone}

ABOUT US:
{value_proposition}

WHAT WE OFFER THEM:
{what_we_offer}

WHAT WE'RE ASKING:
{what_we_ask}

PROSPECT:
- Domain: {prospect_domain}
- Contact: {prospect_name}
- Their page: {prospect_page_title}
- About their page: {prospect_page_summary}
- Domain Rating: {prospect_dr}

RULES:
1. Keep it under 150 words. Busy people don't read long emails.
2. Lead with what's in it for THEM, not what we want.
3. Be specific — reference their site/content to show this isn't a mass email.
4. Suggest ONE clear next step (not multiple options).
5. No begging. This is a collaboration between peers.
6. Sound human, not like a template. No "I hope this finds you well."
7. Sign off as "Ben" (the site owner), not "Ralf" (the agent).

Return JSON:
{{"subject": "...", "body": "...", "follow_up_body": "..."}}

The follow_up_body is a shorter (50-word max) follow-up for if they don't reply in a week.
"""
