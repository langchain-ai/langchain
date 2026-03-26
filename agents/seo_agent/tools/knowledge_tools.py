"""Knowledge base tools — Ralf's SEO/AEO expertise store.

Manages a structured knowledge base in Supabase that Ralf references
when making strategic decisions, writing content, or advising on SEO.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

from agents.seo_agent.tools.supabase_tools import (
    get_client,
    insert_record,
    query_table,
    upsert_record,
)

logger = logging.getLogger(__name__)

# Knowledge categories
CATEGORIES = [
    "technical_seo",
    "content_seo",
    "link_building",
    "on_page_seo",
    "aeo",              # AI Engine Optimisation
    "local_seo",
    "algorithm_updates",
    "tools_and_apis",
    "industry_trends",
]


def store_knowledge(
    category: str,
    topic: str,
    content: str,
    source_urls: list[str] | None = None,
    confidence: str = "high",
) -> dict:
    """Store or update a knowledge entry.

    Args:
        category: One of the CATEGORIES.
        topic: Short topic name (e.g. 'core_web_vitals', 'eeat_guidelines').
        content: The knowledge content (can be multi-paragraph).
        source_urls: URLs where this knowledge was sourced from.
        confidence: 'high', 'medium', or 'low'.
    """
    record = {
        "category": category,
        "topic": topic,
        "content": content,
        "source_urls": source_urls or [],
        "confidence": confidence,
        "last_verified": datetime.now(tz=timezone.utc).isoformat(),
        "updated_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    try:
        return upsert_record("seo_knowledge_base", record, on_conflict="category,topic")
    except Exception:
        logger.warning("Upsert failed, trying insert", exc_info=True)
        return insert_record("seo_knowledge_base", record)


def get_knowledge(category: str | None = None, topic: str | None = None) -> list[dict]:
    """Retrieve knowledge entries.

    Args:
        category: Filter by category. None for all.
        topic: Filter by topic. None for all in category.
    """
    filters = {}
    if category:
        filters["category"] = category
    if topic:
        filters["topic"] = topic

    return query_table("seo_knowledge_base", filters=filters, limit=100)


def get_knowledge_summary() -> str:
    """Get a summary of all knowledge for injection into system prompts."""
    all_entries = query_table("seo_knowledge_base", limit=500)

    if not all_entries:
        return ""

    by_category: dict[str, list[dict]] = {}
    for entry in all_entries:
        cat = entry.get("category", "other")
        by_category.setdefault(cat, []).append(entry)

    summary_parts = ["\n\nSEO/AEO KNOWLEDGE BASE (reference when making decisions):"]
    for cat, entries in sorted(by_category.items()):
        summary_parts.append(f"\n[{cat.upper()}]")
        for e in entries[:5]:  # Limit per category to keep prompt manageable
            content_preview = (e.get("content", "")[:200] + "...") if len(e.get("content", "")) > 200 else e.get("content", "")
            summary_parts.append(f"- {e.get('topic', 'N/A')}: {content_preview}")

    return "\n".join(summary_parts)


def search_knowledge(query: str) -> list[dict]:
    """Search the knowledge base for entries matching a query.

    Simple keyword matching — searches topic and content fields.
    """
    all_entries = query_table("seo_knowledge_base", limit=500)
    query_lower = query.lower()
    matches = []
    for entry in all_entries:
        topic = (entry.get("topic", "") or "").lower()
        content = (entry.get("content", "") or "").lower()
        if query_lower in topic or query_lower in content:
            matches.append(entry)
    return matches
