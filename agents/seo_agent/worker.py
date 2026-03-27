"""Background worker — executes heavy SEO tasks silently.

The worker handles compute-intensive, long-running tasks like content writing,
keyword research, prospect enrichment, and blog publishing. It runs
independently of the pulse and only sends Telegram notifications on completion
or failure — not during execution.

The worker uses the WAL to track progress, the skill registry to decide what
to do, and the gateway for resource management. It is designed to run more
frequently than the old monolithic heartbeat (every 2-3 hours vs 6 hours)
without spamming the user.

Usage::

    python -m agents.seo_agent.worker          # Run once
    python -m agents.seo_agent.worker --loop    # Run on internal schedule
"""

from __future__ import annotations

import html
import logging
import os
import re
import traceback
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


async def execute_worker_cycle() -> dict[str, Any]:
    """Run one worker cycle: evaluate skills, execute tasks via WAL, learn from outcomes.

    This function MUST NOT raise exceptions — all errors are caught and reported.

    Returns:
        Dict with cycle results and tasks executed.
    """
    try:
        return await _execute_worker_inner()
    except Exception:
        logger.error("Worker crashed (contained): %s", traceback.format_exc())
        try:
            from agents.seo_agent.heartbeat import send_telegram

            await send_telegram(f"Worker crashed: {traceback.format_exc()[-300:]}")
        except Exception:
            pass
        return {"status": "crashed", "error": traceback.format_exc()[-500:]}


async def _execute_worker_inner() -> dict[str, Any]:
    """Inner worker logic — may raise exceptions."""
    from agents.seo_agent.gateway import Gateway
    from agents.seo_agent.memory import Memory
    from agents.seo_agent.skills import SkillRegistry
    from agents.seo_agent.tools.crm_tools import get_dashboard_summary
    from agents.seo_agent.wal import WAL

    logger.info("Worker cycle starting...")

    # Boot gateway and get execution context
    gw = Gateway()
    gw.boot()
    ctx = gw.get_execution_context()
    memory = Memory()

    # Check budget
    if ctx.budget_remaining < 0.05:
        logger.info("Budget exhausted (%.1f%% remaining), skipping worker cycle", ctx.budget_remaining * 100)
        return {"status": "budget_exhausted", "budget_remaining": ctx.budget_remaining}

    # Get current state
    try:
        dashboard = get_dashboard_summary()
    except Exception as e:
        logger.error("Dashboard unavailable: %s", e)
        return {"status": "dashboard_error", "error": str(e)}

    # Evaluate skills
    registry = SkillRegistry()
    actionable = registry.evaluate(dashboard, ctx.buffer, budget_remaining=ctx.budget_remaining)

    if not actionable:
        logger.info("No actionable skills this cycle")
        return {"status": "idle", "reason": "no actionable skills"}

    # Plan tasks via WAL
    wal = WAL()
    planned_tasks = []
    for skill, reason in actionable:
        sites = _resolve_sites(skill, ctx.active_sites)
        for site in sites:
            planned_tasks.append({
                "task": skill.name,
                "site": site,
                "reason": reason,
                "skill_priority": skill.priority,
                "cost_tier": skill.cost_tier,
            })

    cycle = wal.begin_cycle(planned_tasks)
    results: list[dict[str, Any]] = []

    # Execute tasks
    for task in cycle.pending_tasks():
        task_name = task["task"]
        site = task.get("site", "all")
        task_id = task["id"]

        logger.info("Worker executing: %s for %s", task_name, site)
        cycle.mark_running(task_id)

        try:
            result = await _execute_skill(task_name, site, ctx, gw, registry)
            summary = _summarise_result(task_name, result)
            cycle.mark_done(task_id, result_summary=summary)
            results.append({"task": task_name, "site": site, "status": "done", "summary": summary})

            # Learn from outcome
            memory.learn_from_outcome(
                task=task_name,
                success=True,
                details=summary,
                site=site,
            )

            # Log activity for recall
            memory.log_activity(
                action_type=task_name,
                summary=summary,
                site=site,
                details=result if isinstance(result, dict) else {},
                source="worker",
            )

            # Record skill execution for cooldown
            skill = registry.get(task_name)
            if skill:
                skill.record_execution(ctx.buffer)

        except Exception as e:
            error_msg = str(e)[:300]
            logger.error("Task %s failed: %s", task_name, error_msg)
            cycle.mark_failed(task_id, error=error_msg)
            results.append({"task": task_name, "site": site, "status": "failed", "error": error_msg})

            # Learn from failure
            memory.learn_from_outcome(
                task=task_name,
                success=False,
                details=error_msg,
                site=site,
            )

    cycle.complete()

    # Send summary notification (only if something was done)
    done_tasks = [r for r in results if r["status"] == "done"]
    failed_tasks = [r for r in results if r["status"] == "failed"]

    if done_tasks or failed_tasks:
        report = _build_worker_report(done_tasks, failed_tasks, ctx)
        try:
            from agents.seo_agent.heartbeat import send_telegram

            await send_telegram(report, parse_mode="HTML")
        except Exception:
            logger.warning("Could not send worker report via Telegram")

    return {"status": "completed", "tasks_done": len(done_tasks), "tasks_failed": len(failed_tasks), "results": results}


async def _execute_skill(
    skill_name: str,
    site: str,
    ctx: Any,
    gw: Any,
    registry: Any,
) -> dict[str, Any]:
    """Execute a single skill, dispatching to the appropriate handler.

    Args:
        skill_name: The skill to execute.
        site: Target site key.
        ctx: Execution context.
        gw: Gateway instance.
        registry: Skill registry.

    Returns:
        Task result dict.
    """
    import asyncio

    skill = registry.get(skill_name)

    # Custom execution for non-graph skills
    if skill_name == "publish_blog":
        return await asyncio.get_event_loop().run_in_executor(
            None, _execute_publish_blog, site, ctx
        )

    if skill_name == "promote_to_crm":
        return await asyncio.get_event_loop().run_in_executor(
            None, _execute_promote_to_crm
        )

    if skill_name == "journal_entry":
        return await asyncio.get_event_loop().run_in_executor(
            None, _execute_journal_entry, ctx
        )

    if skill_name == "memory_consolidation":
        from agents.seo_agent.memory import Memory

        consolidated = Memory().consolidate()
        return {"consolidated": consolidated}

    # Default: execute via gateway (LangGraph task)
    if skill and skill.task_type:
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: gw.execute_task(skill.task_type, site=site)
        )

    msg = f"No executor found for skill: {skill_name}"
    raise ValueError(msg)


def _execute_publish_blog(site: str, ctx: Any) -> dict[str, Any]:
    """Execute the publish_blog skill with keyword selection and diversity checks.

    Args:
        site: Target site key.
        ctx: Execution context.

    Returns:
        Result dict with published post info.
    """
    from agents.seo_agent.tools.supabase_tools import query_table

    # --- Keyword blocklist ---
    _KEYWORD_BLOCKLIST = {
        "b&q", "bq", "b&q kitchen", "bq kitchen", "b&q kitchen units",
        "bq kitchen units", "b and q", "bandq",
    }

    def _is_blocked(kw_text: str) -> bool:
        kw_lower = kw_text.lower().strip()
        for blocked in _KEYWORD_BLOCKLIST:
            if blocked in kw_lower:
                return True
        return False

    # Check memory for additional blocked topics
    memory_blocked: set[str] = set()
    corrections = ctx.memory.recall(category="correction", limit=20)
    for m in corrections:
        content = m.get("content", "").lower()
        if "stop" in content or "avoid" in content or "don't" in content:
            # Extract topic keywords from correction
            words = set(content.split()) - {"stop", "avoid", "don't", "writing", "about", "posts", "the"}
            memory_blocked.update(words)

    # Get existing content to avoid duplicates
    existing_briefs = query_table("seo_content_briefs", limit=500)
    existing_topics = {b.get("keyword", "").lower() for b in existing_briefs}

    existing_slugs: set[str] = set()
    try:
        from agents.seo_agent.tools.github_tools import list_blog_posts, slugify

        existing_posts = list_blog_posts(site)
        existing_slugs = {
            p.get("name", "").replace(".html", "").replace(".mdx", "").replace(".ts", "").lower()
            for p in existing_posts
        }
    except Exception:
        pass

    # Get keywords sorted by volume
    keywords = query_table(
        "seo_keyword_opportunities",
        filters={"target_site": site},
        limit=50,
        order_by="volume",
        order_desc=True,
    )

    # Filter keywords
    untargeted = []
    for k in keywords:
        kw_text = k.get("keyword", "").lower()
        if kw_text in existing_topics:
            continue
        if _is_blocked(kw_text):
            continue
        if any(blocked in kw_text for blocked in memory_blocked):
            continue
        try:
            slug = slugify(kw_text)
        except Exception:
            slug = kw_text.replace(" ", "-")
        if slug in existing_slugs:
            continue
        untargeted.append(k)

    # Topic diversity filter
    _STOP_WORDS = {
        "uk", "free", "online", "best", "how", "to", "a", "the",
        "for", "in", "of", "your", "and", "with", "guide", "ideas", "tips",
    }

    def _significant_words(text: str) -> set:
        return {w for w in text.lower().split() if w not in _STOP_WORDS and len(w) > 2}

    recent_slugs: list[str] = []
    try:
        from agents.seo_agent.tools.github_tools import list_blog_posts

        recent_posts = list_blog_posts(site)
        recent_slugs = [p.get("name", "").replace(".html", "") for p in recent_posts[:5]]
    except Exception:
        pass

    recent_topics = [_significant_words(slug.replace("-", " ")) for slug in recent_slugs[:3]]

    diverse_keywords: list[dict] = []
    for _kw in untargeted:
        kw_words = _significant_words(_kw.get("keyword", ""))
        too_similar = False
        for recent in recent_topics:
            if recent and kw_words:
                overlap = len(kw_words & recent) / max(len(kw_words), 1)
                if overlap > 0.5:
                    too_similar = True
                    break
        if not too_similar:
            diverse_keywords.append(_kw)

    selected = diverse_keywords if diverse_keywords else untargeted

    if not selected:
        return {"status": "no_keywords", "message": "All keywords have content"}

    kw = selected[0]
    kw_text = kw.get("keyword", "")

    from agents.seo_agent.telegram_bot import _generate_blog_post
    from agents.seo_agent.tools.github_tools import publish_blog_post

    blog = _generate_blog_post(site, kw_text, kw_text)
    result = publish_blog_post(
        site=site,
        title=blog["title"],
        content=blog["content"],
        meta_description=blog["meta_description"],
    )

    # Record in buffer to avoid re-selecting same keyword next cycle
    ctx.buffer.set(f"last_blog_keyword_{site}", kw_text, ttl_hours=48)

    return {
        "published": True,
        "title": blog["title"],
        "keyword": kw_text,
        "url": result.get("published_url", "N/A"),
        "site": site,
    }


def _execute_promote_to_crm() -> dict[str, Any]:
    """Promote scored prospects to CRM contacts.

    Returns:
        Result dict with count of promoted prospects.
    """
    from agents.seo_agent.tools.crm_tools import add_crm_contact, get_crm_contacts
    from agents.seo_agent.tools.supabase_tools import query_table

    scored_prospects = query_table(
        "seo_backlink_prospects",
        filters={"status": "scored"},
        limit=20,
    )

    existing_crm = get_crm_contacts(limit=500)
    existing_domains = {
        c.get("website", "").replace("https://", "").replace("http://", "").rstrip("/")
        for c in existing_crm
    }

    promoted = 0
    for p in scored_prospects:
        domain = p.get("domain", "")
        if domain in existing_domains or not domain:
            continue

        segment = p.get("segment", p.get("discovery_method", ""))
        category = "kitchen_company" if "provider" in segment or "company" in segment else "blogger"

        try:
            add_crm_contact(
                company_name=p.get("page_title", domain)[:100] or domain,
                category=category,
                website=f"https://{domain}",
                city="",
                notes=f"Auto-imported from prospect pipeline. DR: {p.get('dr', 0)}. Method: {p.get('discovery_method', '')}",
                source="prospect_pipeline",
                outreach_segment=segment,
            )
            promoted += 1
        except Exception:
            pass

    return {"promoted": promoted}


def _execute_journal_entry(ctx: Any) -> dict[str, Any]:
    """Write and publish a journal entry for Ralf's blog.

    Args:
        ctx: Execution context.

    Returns:
        Result dict with published journal info.
    """
    from agents.seo_agent.tools.github_tools import publish_blog_post
    from agents.seo_agent.tools.reflection_engine import generate_reflective_post

    post = generate_reflective_post()
    result = publish_blog_post(
        site="ralf_seo",
        title=post["title"],
        content=post["content"],
        meta_description=post["meta_description"],
        category=post.get("category", "Field Report"),
        what_i_learned=post.get("what_i_learned", []),
    )

    # Record in buffer
    ctx.buffer.set("last_journal_date", datetime.now(timezone.utc).isoformat(), ttl_hours=120)

    return {
        "published": True,
        "title": post["title"],
        "url": result.get("published_url", "N/A"),
        "category": post.get("category", "Field Report"),
    }


def _resolve_sites(skill: Any, active_sites: dict[str, Any]) -> list[str]:
    """Resolve which sites a skill should run for.

    Args:
        skill: The skill instance.
        active_sites: Dict of active site profiles.

    Returns:
        List of site keys.
    """
    if isinstance(skill.sites, list):
        return [s for s in skill.sites if s in active_sites]
    if skill.sites == "all":
        # Exclude ralf_seo from content/prospecting skills
        if skill.category in ("content", "prospecting") and skill.name != "journal_entry":
            return [s for s in active_sites if s != "ralf_seo" and active_sites[s].get("seed_keywords")]
        return list(active_sites.keys())
    return [skill.sites] if skill.sites in active_sites else []


def _summarise_result(task_name: str, result: dict[str, Any]) -> str:
    """Build a concise summary of a task result.

    Args:
        task_name: The task that was executed.
        result: The raw result dict.

    Returns:
        Human-readable summary string.
    """
    if task_name == "publish_blog":
        if result.get("published"):
            return f"Published: {result.get('title', 'N/A')} ({result.get('url', '')})"
        return result.get("message", "No keywords available")

    if task_name == "promote_to_crm":
        return f"Promoted {result.get('promoted', 0)} prospects to CRM"

    if task_name == "journal_entry":
        return f"Journal: {result.get('title', 'N/A')}"

    if task_name == "memory_consolidation":
        return f"Consolidated {result.get('consolidated', 0)} old memories"

    # Generic summary from graph results
    parts = []
    for key in ("keyword_opportunities", "content_gaps", "backlink_prospects", "scored_prospects"):
        items = result.get(key, [])
        if items:
            parts.append(f"{len(items)} {key.replace('_', ' ')}")

    if result.get("errors"):
        parts.append(f"{len(result['errors'])} errors")

    return ", ".join(parts) if parts else "completed"


_NOOP_PATTERNS = [
    re.compile(r"Promoted 0 prospects"),
    re.compile(r"Consolidated 0 "),
    re.compile(r"^completed$"),
]

_ERROR_REWRITES = [
    (re.compile(r"takes \d+ positional arguments? but \d+ (?:was|were) given"), "Internal routing error"),
    (re.compile(r"rate.?limit|429|too many requests", re.I), "Rate limited \u2014 will retry next cycle"),
    (re.compile(r"timed?\s*out|timeout", re.I), "Request timed out"),
    (re.compile(r"connection.?(?:error|refused|reset)", re.I), "Connection error"),
]

_URL_RE = re.compile(r"\((https?://[^\s)]+)\)")

# Human-readable labels for each task so the report makes sense at a glance.
_TASK_LABELS: dict[str, str] = {
    "publish_blog": "Blog published",
    "keyword_research": "Keyword research",
    "keyword_refresh": "Keyword refresh",
    "content_gap_analysis": "Content gap analysis",
    "discover_prospects": "Find backlink prospects",
    "score_prospects": "Score prospects",
    "promote_to_crm": "Promote to CRM",
    "track_rankings": "Track search rankings",
    "journal_entry": "Journal entry",
    "memory_consolidation": "Memory cleanup",
}


def _is_noop_result(summary: str) -> bool:
    """Check if a task summary indicates nothing meaningful happened.

    Args:
        summary: The task summary string.

    Returns:
        `True` if the result is a no-op.
    """
    return any(p.search(summary) for p in _NOOP_PATTERNS)


def _friendly_error(raw: str) -> str:
    """Map a raw Python error to a user-friendly message.

    Args:
        raw: The raw error string.

    Returns:
        A short, human-readable error description.
    """
    for pattern, friendly in _ERROR_REWRITES:
        if pattern.search(raw):
            return friendly
    # Fallback: take the first line, trimmed, so it's still readable
    first_line = raw.split("\n")[0].strip()[:80]
    return first_line if first_line else "Unknown error"


def _format_html_summary(task: str, summary: str) -> str:
    """Convert a task summary to HTML, turning URLs into clickable links.

    Args:
        task: The task name.
        summary: The raw summary string.

    Returns:
        HTML-formatted summary.
    """
    if task == "publish_blog" and "Published:" in summary:
        match = _URL_RE.search(summary)
        if match:
            url = match.group(1)
            title = summary.split("Published: ")[1].split(" (http")[0]
            return f'Published: <a href="{html.escape(url)}">{html.escape(title)}</a>'
    return html.escape(summary)


def _build_worker_report(
    done: list[dict],
    failed: list[dict],
    ctx: Any,
) -> str:
    """Build an HTML-formatted Telegram worker report.

    Groups successes and failures, filters out no-op tasks, and converts
    raw errors to user-friendly messages.

    Args:
        done: List of completed task results.
        failed: List of failed task results.
        ctx: Execution context.

    Returns:
        HTML-formatted report string for Telegram.
    """
    meaningful = [r for r in done if not _is_noop_result(r["summary"])]
    skipped = len(done) - len(meaningful)
    total = len(done) + len(failed)

    lines: list[str] = [
        f"\U0001f4cb <b>Worker Report</b> \u2014 {total} task(s) ran",
    ]

    if meaningful:
        lines.append("")
        lines.append("\u2705 <b>Completed</b>")
        for r in meaningful:
            label = _TASK_LABELS.get(r["task"], r["task"])
            summary = _format_html_summary(r["task"], r["summary"])
            site = html.escape(r["site"])
            lines.append(f"  \u2022 <b>{label}</b> ({site})")
            lines.append(f"    {summary}")

    if failed:
        lines.append("")
        lines.append("\u274c <b>Failed</b>")
        by_error: dict[str, list[tuple[str, str]]] = {}
        for r in failed:
            friendly = _friendly_error(r["error"])
            label = _TASK_LABELS.get(r["task"], r["task"])
            by_error.setdefault(friendly, []).append(
                (label, html.escape(r["site"]))
            )
        for err, task_pairs in by_error.items():
            # Group tasks sharing the same error on one line
            task_list = ", ".join(
                f"{label} ({site})" for label, site in task_pairs
            )
            lines.append(f"  \u2022 {task_list}")
            lines.append(f"    <i>Reason: {html.escape(err)}</i>")

    if skipped:
        lines.append(f"\n<i>{skipped} task(s) had nothing to do and were skipped.</i>")

    if not meaningful and not failed:
        lines.append("\nNo tasks needed to run this cycle.")

    lines.append(f"\n\U0001f4b0 Budget: {ctx.budget_remaining:.0%} remaining")

    return "\n".join(lines)


def main() -> None:
    """Run the worker."""
    import argparse
    import asyncio

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Ralf SEO worker (background tasks)")
    parser.add_argument("--loop", action="store_true", help="Run on internal schedule")
    parser.add_argument("--interval", type=int, default=180, help="Minutes between runs (default 3 hours)")
    args = parser.parse_args()

    if args.loop:
        import time

        logger.info("Worker loop starting (every %d minutes)...", args.interval)
        while True:
            try:
                asyncio.run(execute_worker_cycle())
            except Exception:
                logger.error("Worker cycle failed: %s", traceback.format_exc())
            logger.info("Sleeping %d minutes until next worker cycle...", args.interval)
            time.sleep(args.interval * 60)
    else:
        asyncio.run(execute_worker_cycle())


if __name__ == "__main__":
    main()
