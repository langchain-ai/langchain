"""SEO-specific prompt templates."""
from __future__ import annotations

from content_growth_agent.prompts.base_prompts import wrap_user_prompt


GEN_TITLES = (
    "Generate 10 catchy, SEO-optimized video titles for the following content. "
    "Include variations for short-form and long-form. Return as a JSON array of strings."
)

GEN_DESCRIPTIONS = (
    "Write 3 SEO-friendly YouTube descriptions (150-250 words) that summarize the video, "
    "include primary keywords and a call-to-action. Return as a JSON array of strings."
)

GEN_KEYWORDS = (
    "Extract 10 relevant keywords and hashtags for SEO from the content. Return as a JSON array."
)


def build_title_prompt(context: str) -> str:
    return wrap_user_prompt(GEN_TITLES, context)


def build_description_prompt(context: str) -> str:
    return wrap_user_prompt(GEN_DESCRIPTIONS, context)


def build_keywords_prompt(context: str) -> str:
    return wrap_user_prompt(GEN_KEYWORDS, context)
