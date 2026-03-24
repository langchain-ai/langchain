"""Internal linker node — suggests internal and cross-site linking opportunities.

Queries existing published pages from Supabase, identifies the most relevant
pages to link to and from, and flags cross-site opportunities between
kitchensdirectory and freeroomplanner.
"""

from __future__ import annotations

import logging
from typing import Any

from agents.seo_agent.config import SITE_PROFILES
from agents.seo_agent.state import SEOAgentState
from agents.seo_agent.tools import file_tools, supabase_tools

logger = logging.getLogger(__name__)

# Cross-site linking pairs
_CROSS_SITE_PAIRS = [
    ("kitchensdirectory", "freeroomplanner"),
    ("freeroomplanner", "kitchensdirectory"),
]

# Number of internal link suggestions per direction
_MAX_LINK_SUGGESTIONS = 5


def _compute_keyword_overlap(
    keywords_a: set[str], keywords_b: set[str]
) -> float:
    """Compute a simple Jaccard-like overlap score between two keyword sets.

    Args:
        keywords_a: First set of keywords.
        keywords_b: Second set of keywords.

    Returns:
        Overlap score between 0.0 and 1.0.
    """
    if not keywords_a or not keywords_b:
        return 0.0
    intersection = keywords_a & keywords_b
    union = keywords_a | keywords_b
    return len(intersection) / len(union)


def _extract_keywords_from_text(text: str) -> set[str]:
    """Extract meaningful words from a text string for overlap comparison.

    Strips common stop words and returns lowercase keyword tokens.

    Args:
        text: The text to extract keywords from.

    Returns:
        A set of lowercase keyword strings.
    """
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "can", "this", "that", "these",
        "those", "it", "its", "your", "our", "their", "my", "his", "her",
        "we", "you", "they", "i", "not", "no", "so", "if", "how", "what",
        "which", "who", "when", "where", "why", "all", "each", "every",
        "both", "few", "more", "most", "other", "some", "such", "than",
        "too", "very", "just", "about",
    }
    words = set()
    for word in text.lower().split():
        cleaned = word.strip(".,!?;:\"'()-/")
        if cleaned and cleaned not in stop_words and len(cleaned) > 2:
            words.add(cleaned)
    return words


def _generate_anchor_variants(keyword: str) -> list[dict[str, str]]:
    """Generate anchor text variants for a keyword.

    Produces exact match, partial match, and natural-sounding anchor texts.

    Args:
        keyword: The target keyword.

    Returns:
        List of dicts with `type` and `text` keys.
    """
    words = keyword.split()
    variants: list[dict[str, str]] = [
        {"type": "exact", "text": keyword},
    ]

    # Partial match — use first 3-4 significant words
    if len(words) > 3:
        variants.append({"type": "partial", "text": " ".join(words[:3])})

    # Natural variants
    natural_templates = [
        f"learn more about {keyword}",
        f"our guide to {keyword}",
        f"find out about {keyword}",
    ]
    if len(words) <= 4:
        natural_templates.append(f"{keyword} guide")

    variants.append({"type": "natural", "text": natural_templates[0]})

    return variants


def run_internal_linker(state: SEOAgentState) -> dict[str, Any]:
    """Suggest internal and cross-site linking opportunities for new content.

    Queries all published pages from Supabase, computes keyword overlap to
    find the most relevant pages to link to and from, identifies cross-site
    opportunities, and appends a linking plan to the content brief file.

    Args:
        state: The current SEO agent state.

    Returns:
        State update with `errors` and `next_node`.
    """
    errors: list[str] = list(state.get("errors", []))
    target_site = state["target_site"]
    brief = state.get("content_brief")
    draft = state.get("content_draft", "")

    if not brief:
        msg = "No content_brief in state — cannot generate linking plan"
        logger.error(msg)
        errors.append(msg)
        return {"errors": errors, "next_node": "END"}

    keyword = brief.get("keyword", "")
    new_content_keywords = _extract_keywords_from_text(keyword)
    if draft:
        # Also extract keywords from the draft title area
        draft_first_lines = "\n".join(draft.split("\n")[:10])
        new_content_keywords |= _extract_keywords_from_text(draft_first_lines)

    # Add semantic keywords from the brief
    for sk in brief.get("semantic_keywords", []):
        new_content_keywords |= _extract_keywords_from_text(sk)

    # Query all published pages across all sites
    published_pages: list[dict[str, Any]] = []
    for site_key in SITE_PROFILES:
        try:
            pages = supabase_tools.query_table(
                "seo_content_drafts",
                filters={"status": "published", "target_site": site_key},
                limit=200,
            )
            for page in pages:
                page["_site_key"] = site_key
            published_pages.extend(pages)
        except Exception:
            msg = f"Failed to query published pages for '{site_key}'"
            logger.warning(msg, exc_info=True)
            errors.append(msg)

    logger.info(
        "Found %d published pages across all sites for linking analysis",
        len(published_pages),
    )

    # Score each published page by keyword overlap with new content
    scored_pages: list[dict[str, Any]] = []
    for page in published_pages:
        page_keywords = _extract_keywords_from_text(
            f"{page.get('keyword', '')} {page.get('title', '')}"
        )
        overlap = _compute_keyword_overlap(new_content_keywords, page_keywords)
        if overlap > 0:
            scored_pages.append({
                "page": page,
                "overlap_score": overlap,
            })

    scored_pages.sort(key=lambda x: x["overlap_score"], reverse=True)

    # Pages to link TO from new content (same site first, then cross-site)
    link_to: list[dict[str, Any]] = []
    for entry in scored_pages[:_MAX_LINK_SUGGESTIONS]:
        page = entry["page"]
        link_to.append({
            "target_page": page.get("title", ""),
            "target_site": page.get("_site_key", ""),
            "file_path": page.get("file_path", ""),
            "keyword": page.get("keyword", ""),
            "overlap_score": round(entry["overlap_score"], 3),
            "anchor_variants": _generate_anchor_variants(
                page.get("keyword", page.get("title", ""))
            ),
            "is_cross_site": page.get("_site_key", "") != target_site,
        })

    # Pages that should link BACK to the new content
    link_from: list[dict[str, Any]] = []
    for entry in scored_pages[:_MAX_LINK_SUGGESTIONS]:
        page = entry["page"]
        link_from.append({
            "source_page": page.get("title", ""),
            "source_site": page.get("_site_key", ""),
            "file_path": page.get("file_path", ""),
            "keyword": page.get("keyword", ""),
            "overlap_score": round(entry["overlap_score"], 3),
            "anchor_variants": _generate_anchor_variants(keyword),
            "is_cross_site": page.get("_site_key", "") != target_site,
        })

    # Flag cross-site linking opportunities specifically
    cross_site_opportunities: list[dict[str, Any]] = []
    for entry in link_to + link_from:
        if entry.get("is_cross_site"):
            cross_site_opportunities.append(entry)

    logger.info(
        "Linking plan for '%s': %d link-to, %d link-from, %d cross-site",
        keyword,
        len(link_to),
        len(link_from),
        len(cross_site_opportunities),
    )

    # Build the linking plan markdown
    linking_plan = _build_linking_plan_markdown(
        keyword=keyword,
        link_to=link_to,
        link_from=link_from,
        cross_site=cross_site_opportunities,
    )

    # Append the linking plan to the brief markdown file
    try:
        existing_brief = file_tools.read_brief(keyword)
        if existing_brief is not None:
            updated_content = existing_brief + "\n\n" + linking_plan
            file_tools.write_brief(keyword, updated_content)
            logger.info("Linking plan appended to brief for '%s'", keyword)
        else:
            logger.warning(
                "No existing brief file found for '%s' — writing standalone",
                keyword,
            )
            file_tools.write_brief(keyword, linking_plan)
    except Exception:
        msg = f"Failed to append linking plan to brief for '{keyword}'"
        logger.warning(msg, exc_info=True)
        errors.append(msg)

    return {"errors": errors, "next_node": "END"}


def _build_linking_plan_markdown(
    keyword: str,
    link_to: list[dict[str, Any]],
    link_from: list[dict[str, Any]],
    cross_site: list[dict[str, Any]],
) -> str:
    """Build a markdown linking plan section.

    Args:
        keyword: The new content's target keyword.
        link_to: Pages to link to from the new content.
        link_from: Pages that should link back to the new content.
        cross_site: Cross-site linking opportunities.

    Returns:
        A markdown-formatted linking plan string.
    """
    lines: list[str] = [
        "---",
        "",
        "## Internal Linking Plan",
        "",
        f"**New content keyword:** {keyword}",
        "",
        "### Pages to Link TO from This Content",
        "",
    ]

    if link_to:
        for entry in link_to:
            site_label = (
                f" [{entry.get('target_site', '')}]"
                if entry.get("is_cross_site") else ""
            )
            lines.append(
                f"- **{entry.get('target_page', 'Untitled')}**{site_label} "
                f"(overlap: {entry.get('overlap_score', 0)})"
            )
            for variant in entry.get("anchor_variants", []):
                lines.append(
                    f"  - {variant.get('type', '')}: \"{variant.get('text', '')}\""
                )
    else:
        lines.append("*No relevant pages found.*")

    lines.extend([
        "",
        "### Pages That Should Link TO This Content",
        "",
    ])

    if link_from:
        for entry in link_from:
            site_label = (
                f" [{entry.get('source_site', '')}]"
                if entry.get("is_cross_site") else ""
            )
            lines.append(
                f"- **{entry.get('source_page', 'Untitled')}**{site_label} "
                f"(overlap: {entry.get('overlap_score', 0)})"
            )
            for variant in entry.get("anchor_variants", []):
                lines.append(
                    f"  - {variant.get('type', '')}: \"{variant.get('text', '')}\""
                )
    else:
        lines.append("*No relevant pages found.*")

    if cross_site:
        lines.extend([
            "",
            "### Cross-Site Linking Opportunities",
            "",
        ])
        for entry in cross_site:
            page_name = entry.get(
                "target_page", entry.get("source_page", "Untitled")
            )
            site = entry.get(
                "target_site", entry.get("source_site", "unknown")
            )
            lines.append(f"- **{page_name}** ({site})")

    return "\n".join(lines) + "\n"
