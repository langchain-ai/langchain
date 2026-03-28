"""Content quality review — autonomous self-check before publishing.

Scores a blog post against a structured rubric covering SEO fundamentals,
originality, concrete detail, voice, depth, and the share test. If the
score is below threshold, attempts a single LLM rewrite pass to fix
flagged issues. Always returns — never blocks publishing.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from agents.seo_agent.tools.llm_router import call_llm

logger = logging.getLogger(__name__)

# Americanisms to flag (subset — the LLM handles the full check)
_AMERICANISMS = [
    "color",
    "center",
    "organization",
    "labor",
    "meter",
    "favorite",
    "realize",
    "optimize",
    "behavior",
    "traveled",
    "catalog",
    "defense",
    "offense",
    "license",  # as a verb is fine, but noun should be "licence" in UK
]

# Filler openers that indicate template-style writing
_FILLER_OPENERS = [
    "In today's world",
    "It goes without saying",
    "In conclusion",
    "Dive into",
    "It's worth noting",
    "Without further ado",
    "When it comes to",
    "Look no further",
    "In this article",
    "Let's explore",
]

_REVIEW_PROMPT = """\
You are a content quality reviewer for UK home improvement blog posts.
Score this post against each dimension below. Be honest and specific.

## Rubric

1. **SEO fundamentals** (0-10)
   - Is the primary keyword in the H1, within the first 100 words, and in at least one H2?
   - Is heading structure logical — one H1, clear H2s, no headings that exist just to break up the page?
   - Does the post hit target word count without padding? Padding = sections restating what was already said.
   - Are there 2-3 internal link opportunities worth noting?
   - Is spelling, punctuation, and phrasing consistently British English?

2. **Originality** (0-10)
   - Is there a point of view, a specific recommendation, or information the reader could NOT get from the first three Google results?
   - Content that agrees with everything and commits to nothing scores low.

3. **Concrete detail** (0-10)
   - Are there specific facts, numbers, prices, timeframes, or trade-offs a homeowner would find useful?
   - Replace vague generalities with specific, useful information.

4. **Voice** (0-10)
   - Does it sound like a knowledgeable person talking, or a content template?
   - Red flags: every sentence starting with "This", filler openers, paragraphs that are just topic sentences with nothing underneath.

5. **Depth** (0-10)
   - Deep enough to be genuinely useful to a homeowner considering a project, without becoming a trade manual?

6. **Share test** (0-10)
   - Would someone who read this send it to a friend planning the same project?

Return ONLY valid JSON with this exact structure (no markdown fences):
{
  "scores": {
    "seo_fundamentals": <int 0-10>,
    "originality": <int 0-10>,
    "concrete_detail": <int 0-10>,
    "voice": <int 0-10>,
    "depth": <int 0-10>,
    "share_test": <int 0-10>
  },
  "overall_score": <int 0-10>,
  "issues": ["<specific issue 1>", "<specific issue 2>", ...],
  "strengths": ["<strength 1>", ...],
  "verdict": "approved" or "needs_revision"
}

A post scoring 7+ overall is "approved". Below 7 is "needs_revision".
"""

_REWRITE_PROMPT = """\
You are a content editor for UK home improvement blog posts.
The following draft was flagged with these issues:

{issues}

Rewrite the content to fix these issues. Preserve the structure, headings,
and overall topic. Focus only on fixing the flagged problems.

Rules:
- British English throughout (colour, centre, organisation, metre)
- No filler phrases or template language
- Add specific details where flagged as vague
- Keep the same heading structure

Return ONLY the revised content in markdown. No preamble or explanation.
"""


def _detect_quick_issues(content: str, keyword: str) -> list[str]:
    """Run fast deterministic checks before the LLM review.

    Args:
        content: The blog post content.
        keyword: The target keyword.

    Returns:
        List of issue strings found.
    """
    issues: list[str] = []

    # Check for Americanisms
    content_lower = content.lower()
    for word in _AMERICANISMS:
        if re.search(rf"\b{word}\b", content_lower):
            issues.append(f"Americanism detected: '{word}' — use British English equivalent")

    # Check for filler openers
    for phrase in _FILLER_OPENERS:
        if phrase.lower() in content_lower:
            issues.append(f"Filler phrase detected: '{phrase}'")

    # Check heading hierarchy
    h1_count = len(re.findall(r"^#\s+", content, re.MULTILINE))
    if h1_count == 0:
        h1_count = len(re.findall(r"<h1[^>]*>", content, re.IGNORECASE))
    if h1_count > 1:
        issues.append(f"Multiple H1 headings found ({h1_count}) — should be exactly one")

    # Check keyword in first 100 words
    if keyword:
        first_100 = " ".join(content.split()[:100]).lower()
        if keyword.lower() not in first_100:
            issues.append(f"Primary keyword '{keyword}' not found in first 100 words")

    # Check for "This" sentence starts (template red flag)
    sentences = re.findall(r"(?:^|\.\s+)(This\s+\w+)", content)
    if len(sentences) > 5:
        issues.append(
            f"Template-style writing: {len(sentences)} sentences start with 'This' — vary sentence openers"
        )

    return issues


def review_before_publish(
    title: str,
    content: str,
    keyword: str = "",
    site: str = "",
    meta_description: str = "",
) -> dict[str, Any]:
    """Run the content quality rubric and attempt fixes if needed.

    This function is designed to be called from ``publish_blog_post()``
    right before the actual commit. It always returns — it never raises
    or blocks publishing.

    Args:
        title: Blog post title.
        content: Blog post content (markdown or HTML).
        keyword: Target SEO keyword.
        site: Site identifier.
        meta_description: Meta description for SEO check.

    Returns:
        Dict with keys: ``content`` (possibly revised), ``score``,
        ``verdict``, ``issues``, ``fixed``, ``review_text``.
    """
    result: dict[str, Any] = {
        "content": content,
        "score": 0,
        "verdict": "approved",
        "issues": [],
        "fixed": [],
        "review_text": "",
    }

    try:
        # Phase 1: Quick deterministic checks
        quick_issues = _detect_quick_issues(content, keyword)

        # Phase 2: LLM rubric scoring
        word_count = len(content.split())
        meta_note = ""
        if meta_description:
            meta_len = len(meta_description)
            meta_note = f"\nMeta description ({meta_len} chars): {meta_description}"

        review_message = (
            f"Primary keyword: {keyword}\n"
            f"Title: {title}\n"
            f"Word count: {word_count}{meta_note}\n\n"
            f"Content:\n{content[:4000]}"
        )

        llm_result = call_llm(
            task="review_blog_post",
            messages=[{"role": "user", "content": review_message}],
            system=_REVIEW_PROMPT,
            site=site,
        )

        raw_text = llm_result.get("text", "").strip()
        # Strip markdown fences if present
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1]
        if raw_text.endswith("```"):
            raw_text = raw_text.rsplit("```", 1)[0]

        review = json.loads(raw_text.strip())
        score = int(review.get("overall_score", 7))
        llm_issues = review.get("issues", [])
        verdict = review.get("verdict", "approved")

        all_issues = quick_issues + llm_issues
        result["score"] = score
        result["verdict"] = verdict
        result["issues"] = all_issues
        result["review_text"] = raw_text

        logger.info(
            "Content review for '%s': score=%d, verdict=%s, issues=%d",
            title, score, verdict, len(all_issues),
        )

        # Phase 3: Attempt rewrite if needs revision and score >= 4
        if verdict == "needs_revision" and score >= 4 and all_issues:
            logger.info("Attempting auto-fix rewrite for '%s'", title)
            issues_text = "\n".join(f"- {issue}" for issue in all_issues)

            rewrite_result = call_llm(
                task="write_blog_post",
                messages=[
                    {
                        "role": "user",
                        "content": _REWRITE_PROMPT.format(issues=issues_text)
                        + f"\n\nOriginal content:\n{content[:4000]}",
                    }
                ],
                site=site,
            )

            revised = rewrite_result.get("text", "").strip()
            if revised and len(revised) > len(content) * 0.5:
                result["content"] = revised
                result["fixed"] = all_issues
                result["verdict"] = "approved_after_revision"
                logger.info("Auto-fix rewrite applied for '%s'", title)
            else:
                logger.warning("Auto-fix rewrite too short or empty, keeping original")

    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("Content review JSON parse failed: %s", exc)
        result["issues"] = quick_issues if "quick_issues" in dir() else []
        result["verdict"] = "review_skipped"
    except Exception:
        logger.warning("Content review failed, proceeding with original", exc_info=True)
        result["verdict"] = "review_skipped"

    return result
