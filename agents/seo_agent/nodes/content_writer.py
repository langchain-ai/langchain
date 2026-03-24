"""Content writer node — drafts SEO content using a two-step LLM pipeline.

Implements a Draft-then-Refine pattern:
1. Haiku generates a structured outline from the brief.
2. Sonnet writes each section, producing the full draft.

Enforces UK English, avoids filler phrases, and includes site-specific
customisations (maker references for kitchensdirectory, CTAs for freeroomplanner).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agents.seo_agent.config import SITE_PROFILES
from agents.seo_agent.state import SEOAgentState
from agents.seo_agent.tools import file_tools, supabase_tools
from agents.seo_agent.tools.llm_router import call_llm

logger = logging.getLogger(__name__)

# Filler phrases that must not appear in output
_BANNED_PHRASES = [
    "In conclusion",
    "Dive into",
    "It's worth noting",
    "It is worth noting",
    "Without further ado",
    "In today's world",
    "At the end of the day",
    "When it comes to",
    "Look no further",
    "In this article",
    "Let's explore",
]

_TONE_RULES = (
    "## Writing Rules\n"
    "- Use UK English throughout (e.g. 'colour' not 'color', 'organisation' "
    "not 'organization', 'centre' not 'center').\n"
    "- No Americanisms.\n"
    "- NEVER use these filler phrases: "
    + ", ".join(f'"{p}"' for p in _BANNED_PHRASES)
    + ".\n"
    "- Tone: conversational but authoritative. Write as a knowledgeable friend, "
    "not a salesperson.\n"
    "- Use short paragraphs (2-4 sentences max).\n"
    "- Include specific data, examples, and practical advice where possible.\n"
)


def _build_cached_system(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a system prompt with prompt caching for site profile and tone rules.

    Args:
        profile: The site profile dict from config.

    Returns:
        A list of content blocks for the Anthropic `system` parameter.
    """
    site_profile_text = json.dumps(profile, indent=2)

    return [
        {
            "type": "text",
            "text": (
                "You are an expert SEO content writer specialising in UK home "
                "improvement and renovation topics.\n\n"
                "## Site Profile\n"
                f"```json\n{site_profile_text}\n```\n\n"
                f"{_TONE_RULES}"
            ),
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": (
                "Write content that is genuinely useful to readers. "
                "Every section should deliver value — facts, comparisons, "
                "actionable steps, or real examples. Avoid padding."
            ),
        },
    ]


def _generate_outline(
    brief: dict[str, Any],
    weekly_spend: float,
    target_site: str,
) -> list[dict[str, Any]]:
    """Use Haiku to generate a structured outline from the content brief.

    Args:
        brief: The content brief dict.
        weekly_spend: Current weekly LLM spend in USD.
        target_site: The target site identifier.

    Returns:
        A list of section dicts with `heading` and `instructions` keys.
    """
    brief_summary = json.dumps(
        {
            k: brief.get(k)
            for k in (
                "title",
                "headings",
                "semantic_keywords",
                "faq_questions",
                "cta",
                "content_type",
                "target_word_count",
            )
        },
        indent=2,
    )

    messages = [
        {
            "role": "user",
            "content": (
                "Given this content brief, produce a detailed writing outline. "
                "For each section, include the heading and specific instructions "
                "for what to cover.\n\n"
                f"Brief:\n```json\n{brief_summary}\n```\n\n"
                "Return a JSON array of objects, each with:\n"
                '- "heading": the section heading\n'
                '- "instructions": what the writer should cover in this section\n'
                '- "approx_words": target word count for this section\n\n'
                "Return ONLY the JSON array, no markdown fences."
            ),
        }
    ]

    result = call_llm(
        task="filter_keywords",
        messages=messages,
        weekly_spend=weekly_spend,
        site=target_site,
        log_fn=supabase_tools.log_llm_cost,
    )

    raw_text = result.get("text", "").strip()
    # Strip any markdown fences
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[1]
    if raw_text.endswith("```"):
        raw_text = raw_text.rsplit("```", 1)[0]

    try:
        outline = json.loads(raw_text.strip())
        if isinstance(outline, list):
            return outline
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse outline JSON, using brief headings as fallback")

    # Fallback: derive outline from brief headings
    return [
        {
            "heading": h.get("text", f"Section {i + 1}"),
            "instructions": f"Write about: {h.get('text', '')}",
            "approx_words": 300,
        }
        for i, h in enumerate(brief.get("headings", []))
    ]


def _write_section(
    section: dict[str, Any],
    brief: dict[str, Any],
    system_prompt: list[dict[str, Any]],
    weekly_spend: float,
    target_site: str,
    site_context: str,
) -> dict[str, Any]:
    """Write a single section of the article using Sonnet.

    Args:
        section: The outline section dict with heading and instructions.
        brief: The full content brief.
        system_prompt: Cached system prompt blocks.
        weekly_spend: Current weekly LLM spend.
        target_site: The target site identifier.
        site_context: Extra site-specific context to inject.

    Returns:
        Dict with `heading`, `content`, and `cost_usd` keys.
    """
    keyword = brief.get("keyword", "")
    semantic_keywords = ", ".join(brief.get("semantic_keywords", [])[:10])

    user_prompt = (
        f"Write the following section of an article about \"{keyword}\".\n\n"
        f"## Section: {section.get('heading', '')}\n"
        f"Instructions: {section.get('instructions', '')}\n"
        f"Target length: ~{section.get('approx_words', 300)} words\n"
        f"Semantic keywords to weave in naturally: {semantic_keywords}\n"
    )

    if site_context:
        user_prompt += f"\n{site_context}\n"

    user_prompt += (
        "\nWrite ONLY the section content in markdown. "
        "Start with the section heading as an H2. "
        "Do not include a title or any preamble."
    )

    messages = [{"role": "user", "content": user_prompt}]

    result = call_llm(
        task="write_blog_post",
        messages=messages,
        system=system_prompt,
        weekly_spend=weekly_spend,
        site=target_site,
        log_fn=supabase_tools.log_llm_cost,
    )

    return {
        "heading": section.get("heading", ""),
        "content": result.get("text", ""),
        "cost_usd": result.get("cost_usd", 0.0),
    }


def _get_site_context(target_site: str, brief: dict[str, Any]) -> str:
    """Build site-specific context to inject into writing prompts.

    Args:
        target_site: The site identifier.
        brief: The content brief dict.

    Returns:
        A context string, or empty string if no special context applies.
    """
    if target_site == "kitchensdirectory":
        # Reference real makers from Supabase
        keyword = brief.get("keyword", "")
        # Extract a city name from the keyword if present
        city_candidates = [
            word for word in keyword.split()
            if word[0].isupper() and word.lower() not in (
                "uk", "the", "best", "top", "how", "what", "kitchen",
                "kitchens", "makers", "companies",
            )
        ] if keyword else []

        if city_candidates:
            city = city_candidates[0]
            try:
                makers = supabase_tools.get_makers_by_location(city)
                if makers:
                    maker_names = [m.get("name", "") for m in makers[:5]]
                    return (
                        f"Reference these real kitchen makers in {city} where "
                        f"relevant: {', '.join(maker_names)}. "
                        f"Mention them naturally as examples, not as a list dump."
                    )
            except Exception:
                logger.warning(
                    "Failed to fetch makers for %s", city, exc_info=True
                )

    if target_site == "freeroomplanner":
        return (
            "Include natural calls-to-action encouraging readers to try the "
            "free room planner tool. For example: 'Try our free room planner "
            "to visualise your layout before committing to a design.' "
            "Integrate these CTAs within relevant sections, not just at the end."
        )

    return ""


def _generate_self_critique(
    full_draft: str,
    brief: dict[str, Any],
    system_prompt: list[dict[str, Any]],
    weekly_spend: float,
    target_site: str,
) -> str:
    """Generate a self-critique of the draft against the brief.

    Args:
        full_draft: The assembled draft markdown.
        brief: The content brief.
        system_prompt: Cached system prompt blocks.
        weekly_spend: Current weekly LLM spend.
        target_site: The target site identifier.

    Returns:
        Markdown self-critique text.
    """
    keyword = brief.get("keyword", "")
    messages = [
        {
            "role": "user",
            "content": (
                f"Review this draft for the keyword \"{keyword}\" against "
                f"the brief and provide a self-critique.\n\n"
                f"Brief title: {brief.get('title', '')}\n"
                f"Target word count: {brief.get('target_word_count', 'N/A')}\n"
                f"Required semantic keywords: "
                f"{', '.join(brief.get('semantic_keywords', []))}\n"
                f"Required FAQs: {', '.join(brief.get('faq_questions', []))}\n\n"
                f"Draft:\n{full_draft[:3000]}\n\n"
                "Provide a brief critique covering:\n"
                "1. Keyword coverage — are semantic keywords woven in?\n"
                "2. Structure — does it follow the brief outline?\n"
                "3. Tone — is it conversational but authoritative?\n"
                "4. UK English — any Americanisms slipped through?\n"
                "5. Missing elements — any brief requirements not addressed?\n"
                "6. Suggested improvements\n\n"
                "Keep it concise — bullet points preferred."
            ),
        }
    ]

    try:
        result = call_llm(
            task="filter_keywords",
            messages=messages,
            system=system_prompt,
            weekly_spend=weekly_spend,
            site=target_site,
            log_fn=supabase_tools.log_llm_cost,
        )
        return result.get("text", "")
    except Exception:
        logger.warning("Self-critique generation failed", exc_info=True)
        return "*Self-critique generation failed.*"


def run_content_writer(state: SEOAgentState) -> dict[str, Any]:
    """Draft SEO content using a two-step outline-then-write pipeline.

    Step 1 (Haiku): Generates a structured outline from the content brief.
    Step 2 (Sonnet): Writes each section individually, then assembles the draft.
    Appends a self-critique section and persists metadata to Supabase.

    Args:
        state: The current SEO agent state.

    Returns:
        State update with `content_draft`, `errors`, and `next_node`.
    """
    errors: list[str] = list(state.get("errors", []))
    target_site = state["target_site"]
    brief = state.get("content_brief")

    if not brief:
        msg = "No content_brief in state — cannot write content"
        logger.error(msg)
        errors.append(msg)
        return {
            "content_draft": None,
            "errors": errors,
            "next_node": "END",
        }

    profile = SITE_PROFILES.get(target_site)
    if profile is None:
        msg = f"No site profile found for '{target_site}'"
        logger.error(msg)
        errors.append(msg)
        return {
            "content_draft": None,
            "errors": errors,
            "next_node": "END",
        }

    keyword = brief.get("keyword", "")
    weekly_spend = state.get("llm_spend_this_week", 0.0)
    total_cost = 0.0

    # Build cached system prompt
    system_prompt = _build_cached_system(profile)

    # Step 1: Generate outline using Haiku
    logger.info("Generating outline for '%s'", keyword)
    try:
        outline = _generate_outline(brief, weekly_spend, target_site)
    except Exception:
        msg = f"Outline generation failed for '{keyword}'"
        logger.error(msg, exc_info=True)
        errors.append(msg)
        return {
            "content_draft": None,
            "errors": errors,
            "next_node": "END",
        }

    logger.info("Outline has %d sections", len(outline))

    # Get site-specific context
    site_context = _get_site_context(target_site, brief)

    # Step 2: Write each section using Sonnet
    sections: list[str] = []
    title = brief.get("title", keyword)
    sections.append(f"# {title}\n")

    # Add meta description as a comment for the editor
    meta_desc = brief.get("meta_description", "")
    if meta_desc:
        sections.append(f"<!-- Meta: {meta_desc} -->\n")

    for i, section in enumerate(outline):
        logger.info(
            "Writing section %d/%d: %s",
            i + 1,
            len(outline),
            section.get("heading", ""),
        )
        try:
            result = _write_section(
                section=section,
                brief=brief,
                system_prompt=system_prompt,
                weekly_spend=weekly_spend + total_cost,
                target_site=target_site,
                site_context=site_context,
            )
            sections.append(result["content"])
            total_cost += result.get("cost_usd", 0.0)
        except Exception:
            msg = (
                f"Failed to write section '{section.get('heading', '')}' "
                f"for '{keyword}'"
            )
            logger.warning(msg, exc_info=True)
            errors.append(msg)
            sections.append(
                f"## {section.get('heading', 'Section')}\n\n"
                f"*[Section writing failed — manual completion required]*\n"
            )

    # Assemble the full draft
    full_draft = "\n\n".join(sections)

    # Generate and append self-critique
    logger.info("Generating self-critique for '%s'", keyword)
    critique = _generate_self_critique(
        full_draft=full_draft,
        brief=brief,
        system_prompt=system_prompt,
        weekly_spend=weekly_spend + total_cost,
        target_site=target_site,
    )
    full_draft += "\n\n---\n\n## Self-Critique\n\n" + critique

    # Write the draft to file
    try:
        file_path = file_tools.write_draft(keyword, full_draft)
        logger.info("Draft written to %s", file_path)
    except Exception:
        file_path = ""
        msg = f"Failed to write draft file for '{keyword}'"
        logger.warning(msg, exc_info=True)
        errors.append(msg)

    # Save metadata to Supabase
    word_count = len(full_draft.split())
    try:
        supabase_tools.insert_record(
            "seo_content_drafts",
            {
                "keyword": keyword,
                "target_site": target_site,
                "title": title,
                "word_count": word_count,
                "file_path": file_path,
                "self_critique": critique,
                "status": "draft",
            },
        )
    except Exception:
        msg = f"Failed to save draft metadata to Supabase for '{keyword}'"
        logger.warning(msg, exc_info=True)
        errors.append(msg)

    # Log total cost for this writing session
    logger.info(
        "Content draft complete for '%s': %d words, $%.4f total cost",
        keyword,
        word_count,
        total_cost,
    )

    return {
        "content_draft": full_draft,
        "errors": errors,
        "next_node": "internal_linker",
    }
