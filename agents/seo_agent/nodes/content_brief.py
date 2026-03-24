"""Content brief node — generates a structured SEO content brief via LLM.

Takes a selected keyword, builds a prompt with cached site profile context,
and produces a brief containing title, headings, FAQs, internal links, and more.
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


def _build_system_prompt(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a system prompt with prompt caching for the static site profile.

    The site profile block is marked with `cache_control` so that repeated
    calls for the same site reuse cached tokens.

    Args:
        profile: The site profile dict from config.

    Returns:
        A list of content blocks suitable for the Anthropic `system` parameter.
    """
    site_profile_text = json.dumps(profile, indent=2)

    return [
        {
            "type": "text",
            "text": (
                "You are an expert SEO content strategist. "
                "Your job is to produce structured content briefs that a writer "
                "can follow to create high-ranking, useful content.\n\n"
                "## Site Profile\n"
                f"```json\n{site_profile_text}\n```"
            ),
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": (
                "## Brief Requirements\n"
                "Produce a JSON object with exactly these keys:\n"
                "- title: SEO-optimised H1 title\n"
                "- meta_description: Under 155 characters, includes the keyword\n"
                "- target_word_count: Recommended word count (integer)\n"
                "- content_type: e.g. 'buyer_guide', 'how_to', 'listicle', "
                "'location_page', 'comparison'\n"
                "- headings: Array of {level: int, text: string} for H2/H3 outline\n"
                "- semantic_keywords: Array of related keywords to weave in\n"
                "- faq_questions: Array of FAQ questions for a schema-ready FAQ section\n"
                "- internal_links: Array of {anchor: string, suggested_path: string}\n"
                "- cta: The primary call-to-action for this content\n\n"
                "Return ONLY the JSON object, no markdown fences or commentary."
            ),
        },
    ]


def run_content_brief(state: SEOAgentState) -> dict[str, Any]:
    """Generate a structured content brief for the selected keyword.

    Calls the LLM with a cached site profile system prompt and the target
    keyword. The resulting brief is written to a markdown file and saved
    to the `seo_content_briefs` Supabase table.

    Args:
        state: The current SEO agent state.

    Returns:
        State update with `content_brief`, `errors`, and `next_node`.
    """
    errors: list[str] = list(state.get("errors", []))
    target_site = state["target_site"]
    selected_keyword = state.get("selected_keyword")

    if not selected_keyword:
        msg = "No selected_keyword in state — cannot generate brief"
        logger.error(msg)
        errors.append(msg)
        return {
            "content_brief": None,
            "errors": errors,
            "next_node": "END",
        }

    profile = SITE_PROFILES.get(target_site)
    if profile is None:
        msg = f"No site profile found for '{target_site}'"
        logger.error(msg)
        errors.append(msg)
        return {
            "content_brief": None,
            "errors": errors,
            "next_node": "END",
        }

    # Build the system prompt with prompt caching on the site profile block
    system_prompt = _build_system_prompt(profile)

    messages = [
        {
            "role": "user",
            "content": (
                f"Create a comprehensive SEO content brief for the keyword: "
                f'"{selected_keyword}"\n\n'
                f"Target site: {profile.get('domain', target_site)}\n"
                f"Target audience: {profile.get('target_audience', 'general')}\n"
                f"Geo focus: {profile.get('geo_focus', 'UK')}"
            ),
        }
    ]

    # Call the LLM
    try:
        llm_result = call_llm(
            task="write_content_brief",
            messages=messages,
            system=system_prompt,
            weekly_spend=state.get("llm_spend_this_week", 0.0),
            site=target_site,
            log_fn=supabase_tools.log_llm_cost,
        )
    except Exception:
        msg = f"LLM call failed for content brief on '{selected_keyword}'"
        logger.error(msg, exc_info=True)
        errors.append(msg)
        return {
            "content_brief": None,
            "errors": errors,
            "next_node": "END",
        }

    # Parse the JSON brief from LLM output
    raw_text = llm_result.get("text", "")
    try:
        # Strip any markdown fences the model might have included
        clean_text = raw_text.strip()
        if clean_text.startswith("```"):
            clean_text = clean_text.split("\n", 1)[1]
        if clean_text.endswith("```"):
            clean_text = clean_text.rsplit("```", 1)[0]
        brief: dict[str, Any] = json.loads(clean_text.strip())
    except (json.JSONDecodeError, IndexError):
        msg = f"Failed to parse LLM output as JSON for '{selected_keyword}'"
        logger.error(msg)
        logger.debug("Raw LLM output: %s", raw_text[:500])
        errors.append(msg)
        # Store the raw text as a fallback
        brief = {
            "title": selected_keyword,
            "meta_description": "",
            "target_word_count": 1500,
            "content_type": "article",
            "headings": [],
            "semantic_keywords": [],
            "faq_questions": [],
            "internal_links": [],
            "cta": "",
            "raw_llm_output": raw_text,
        }

    # Ensure the keyword is attached to the brief
    brief["keyword"] = selected_keyword
    brief["target_site"] = target_site

    # Write the brief to a markdown file
    brief_markdown = _brief_to_markdown(brief, selected_keyword)
    try:
        file_path = file_tools.write_brief(selected_keyword, brief_markdown)
        brief["file_path"] = file_path
        logger.info("Brief written to %s", file_path)
    except Exception:
        msg = f"Failed to write brief file for '{selected_keyword}'"
        logger.warning(msg, exc_info=True)
        errors.append(msg)

    # Save to Supabase
    try:
        supabase_tools.insert_record(
            "seo_content_briefs",
            {
                "keyword": selected_keyword,
                "target_site": target_site,
                "content_type": brief.get("content_type", ""),
                "title": brief.get("title", ""),
                "meta_description": brief.get("meta_description", ""),
                "target_word_count": brief.get("target_word_count", 0),
                "headings": brief.get("headings", []),
                "semantic_keywords": brief.get("semantic_keywords", []),
                "faq_questions": brief.get("faq_questions", []),
                "internal_links": brief.get("internal_links", []),
                "cta": brief.get("cta", ""),
                "brief_json": brief,
                "file_path": brief.get("file_path", ""),
            },
        )
    except Exception:
        msg = f"Failed to save brief to Supabase for '{selected_keyword}'"
        logger.warning(msg, exc_info=True)
        errors.append(msg)

    logger.info(
        "Content brief generated for '%s' (cost: $%.4f)",
        selected_keyword,
        llm_result.get("cost_usd", 0.0),
    )

    return {
        "content_brief": brief,
        "errors": errors,
        "next_node": "content_writer",
    }


def _brief_to_markdown(brief: dict[str, Any], keyword: str) -> str:
    """Convert a structured brief dict to readable markdown.

    Args:
        brief: The brief dictionary.
        keyword: The target keyword.

    Returns:
        A markdown-formatted string.
    """
    lines: list[str] = [
        f"# Content Brief: {brief.get('title', keyword)}",
        "",
        f"**Keyword:** {keyword}",
        f"**Content Type:** {brief.get('content_type', 'N/A')}",
        f"**Target Word Count:** {brief.get('target_word_count', 'N/A')}",
        f"**Meta Description:** {brief.get('meta_description', 'N/A')}",
        f"**CTA:** {brief.get('cta', 'N/A')}",
        "",
        "## Outline",
        "",
    ]

    for heading in brief.get("headings", []):
        level = heading.get("level", 2)
        prefix = "#" * level
        lines.append(f"{prefix} {heading.get('text', '')}")

    lines.extend(["", "## Semantic Keywords", ""])
    for sk in brief.get("semantic_keywords", []):
        lines.append(f"- {sk}")

    lines.extend(["", "## FAQ Questions", ""])
    for faq in brief.get("faq_questions", []):
        lines.append(f"- {faq}")

    lines.extend(["", "## Internal Links", ""])
    for link in brief.get("internal_links", []):
        anchor = link.get("anchor", "")
        path = link.get("suggested_path", "")
        lines.append(f"- [{anchor}]({path})")

    return "\n".join(lines) + "\n"
