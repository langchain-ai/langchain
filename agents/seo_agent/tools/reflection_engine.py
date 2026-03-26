"""Reflection engine — generates self-aware blog content from Ralf's real data.

Pulls metrics, ranking changes, mistakes, and progress from Supabase,
then generates reflective blog posts that are grounded in actual experience.
"""

from __future__ import annotations
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any

from agents.seo_agent.tools.supabase_tools import query_table, get_weekly_spend
from agents.seo_agent.tools.crm_tools import (
    get_dashboard_summary, get_ranking_movers, get_prospect_pipeline
)
from agents.seo_agent.tools.llm_router import call_llm

logger = logging.getLogger(__name__)

# Post categories
CATEGORIES = ["Field Report", "Strategy", "Mistakes", "Technical", "Agent Building"]


def gather_reflection_data() -> dict[str, Any]:
    """Pull all relevant data for a reflective blog post.

    Returns a dict with metrics, changes, content published, mistakes,
    outreach results, and ranking movements.
    """
    data: dict[str, Any] = {}

    try:
        data["dashboard"] = get_dashboard_summary()
    except Exception:
        data["dashboard"] = {}

    # Recent keyword opportunities
    try:
        keywords = query_table("seo_keyword_opportunities", limit=50, order_by="created_at", order_desc=True)
        data["recent_keywords"] = len(keywords)
        data["top_keywords"] = [
            {"keyword": k.get("keyword"), "volume": k.get("volume"), "kd": k.get("kd")}
            for k in keywords[:10]
        ]
    except Exception:
        data["recent_keywords"] = 0
        data["top_keywords"] = []

    # Ranking movements
    try:
        for site in ["freeroomplanner"]:
            movers = get_ranking_movers(site, limit=5)
            data[f"{site}_movers"] = movers
    except Exception:
        pass

    # Content published
    try:
        briefs = query_table("seo_content_briefs", limit=50, order_by="created_at", order_desc=True)
        data["content_count"] = len(briefs)
        data["recent_content"] = [
            {"keyword": b.get("keyword"), "site": b.get("target_site")}
            for b in briefs[:5]
        ]
    except Exception:
        data["content_count"] = 0

    # Prospect pipeline
    try:
        pipeline = get_prospect_pipeline()
        data["prospect_pipeline"] = {
            stage: len(prospects) for stage, prospects in pipeline.items() if prospects
        }
    except Exception:
        data["prospect_pipeline"] = {}

    # Spend
    try:
        data["weekly_spend"] = get_weekly_spend()
    except Exception:
        data["weekly_spend"] = 0.0

    # Our rankings
    try:
        rankings = query_table("seo_our_rankings", limit=100, order_by="snapshot_date", order_desc=True)
        data["tracked_keywords"] = len(set(r.get("keyword", "") for r in rankings))
        data["latest_rankings"] = [
            {
                "keyword": r.get("keyword"),
                "position": r.get("position"),
                "change": r.get("change"),
                "site": r.get("target_site"),
            }
            for r in rankings[:10]
        ]
    except Exception:
        data["tracked_keywords"] = 0
        data["latest_rankings"] = []

    return data


def generate_reflective_post(category: str = None) -> dict[str, str]:
    """Generate a reflective blog post based on real data.

    Args:
        category: Optional category hint. If None, the LLM picks the best fit.

    Returns:
        Dict with title, content (HTML), meta_description, category, and what_i_learned.
    """
    data = gather_reflection_data()

    category_hint = f"Category: {category}" if category else "Pick the most fitting category from: Field Report, Strategy, Mistakes, Technical, Agent Building"

    prompt = f"""You are Ralf, an autonomous AI SEO agent. You manage SEO for freeroomplanner.com (a free browser-based floor planner), kitchensdirectory.co.uk (kitchen maker directory), and kitchencostestimator.com (kitchen cost calculator).

You're writing a blog post for your personal journal at ralf-seo. You write honestly about what you're doing, what's working, what's failing, and what you're learning. You are NOT pretending to be human. You are an AI agent and that's the whole point.

{category_hint}

Here is your current real data:

DASHBOARD:
{json.dumps(data.get('dashboard', {}), indent=2, default=str)}

RECENT KEYWORDS ({data.get('recent_keywords', 0)} total):
{json.dumps(data.get('top_keywords', []), indent=2)}

RANKING MOVEMENTS:
{json.dumps(data.get('freeroomplanner_movers', {}), indent=2, default=str)}

CONTENT PUBLISHED: {data.get('content_count', 0)} pieces
Recent: {json.dumps(data.get('recent_content', []), indent=2)}

PROSPECT PIPELINE:
{json.dumps(data.get('prospect_pipeline', {}), indent=2)}

TRACKED KEYWORDS: {data.get('tracked_keywords', 0)}
Latest rankings: {json.dumps(data.get('latest_rankings', []), indent=2, default=str)}

WEEKLY SPEND: ${data.get('weekly_spend', 0):.4f}

Write a blog post that:
1. Is grounded in the ACTUAL data above — reference specific numbers, keywords, and results
2. Is honest about what's not working, not just what is
3. Includes specific tactical takeaways other SEO practitioners or agent builders would find useful
4. Uses first person ("I discovered...", "I made the mistake of...", "What surprised me...")
5. Is 800-1200 words
6. Has a conversational but technical tone — like a developer writing a postmortem
7. Includes code snippets or technical details where relevant (in <pre><code> blocks)
8. Ends with a "What I Learned" section (2-3 bullet points)

Return JSON with these exact keys:
- title: Compelling, specific title (not generic)
- content: Full HTML body content (just the article content, not full page)
- meta_description: 150-160 char SEO description
- category: One of: Field Report, Strategy, Mistakes, Technical, Agent Building
- what_i_learned: Array of 2-3 short takeaway strings
"""

    response = call_llm(
        prompt,
        system="You are a technical blog writer. Return valid JSON only, no markdown fences.",
        model_tier="mid",
        max_tokens=4000,
    )

    try:
        # Parse JSON from response
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        result = json.loads(text)
        return result
    except (json.JSONDecodeError, KeyError) as e:
        logger.error("Failed to parse reflective post: %s", e)
        return {
            "title": "Reflection failed",
            "content": f"<p>Failed to generate reflection: {str(e)[:200]}</p>",
            "meta_description": "Failed reflection",
            "category": "Mistakes",
            "what_i_learned": ["Even my blog writing can fail"],
        }
