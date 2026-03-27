"""Ralf's heartbeat — orchestrates the Worker and Pulse subsystems.

The heartbeat is the top-level scheduler that coordinates:
- **Worker**: Heavy background tasks (content writing, keyword research, prospecting)
- **Pulse**: Lightweight check-ins (ranking movers, budget alerts, progress summaries)

Both subsystems use the WAL for crash recovery, the skill registry for dynamic
task selection, episodic memory for learning, and the gateway for resource management.

The heartbeat can also run in legacy mode (``--legacy``) which uses the original
monolithic decision tree for backward compatibility.

Usage::

    python -m agents.seo_agent.heartbeat              # Worker + Pulse (default)
    python -m agents.seo_agent.heartbeat --worker      # Worker only
    python -m agents.seo_agent.heartbeat --pulse        # Pulse only
    python -m agents.seo_agent.heartbeat --legacy       # Legacy monolithic mode
    python -m agents.seo_agent.heartbeat --loop         # Run on internal schedule
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import traceback
from datetime import datetime, timezone

from dotenv import load_dotenv

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("heartbeat")

# Telegram chat ID for the owner (Ben)
OWNER_CHAT_ID = int(os.getenv("TELEGRAM_OWNER_CHAT_ID", "7428463356"))


async def send_telegram(message: str, parse_mode: str = "") -> None:
    """Send a message to the owner via Telegram."""
    token = "".join(os.environ.get("TELEGRAM_BOT_TOKEN", "").split())
    if not token:
        logger.error("No TELEGRAM_BOT_TOKEN set")
        return

    import httpx

    async with httpx.AsyncClient() as client:
        payload = {"chat_id": OWNER_CHAT_ID, "text": message}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        try:
            resp = await client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json=payload,
                timeout=15,
            )
            if resp.status_code != 200:
                logger.error("Telegram send failed: %s", resp.text[:200])
        except Exception as e:
            logger.error("Telegram send error: %s", e)


def run_task(task_type: str, **kwargs) -> dict:
    """Run an SEO agent task synchronously (legacy helper, still used by telegram_bot).

    Args:
        task_type: The task to run.
        **kwargs: Additional task parameters.

    Returns:
        The task result dict.
    """
    from agents.seo_agent.agent import build_graph, create_initial_state
    from agents.seo_agent.tools.supabase_tools import ensure_tables, get_weekly_spend

    ensure_tables()
    weekly_spend = get_weekly_spend()

    state = create_initial_state(task_type=task_type, **kwargs)
    state["llm_spend_this_week"] = weekly_spend

    graph = build_graph()
    return graph.invoke(state)


# ---------------------------------------------------------------------------
# New architecture: Worker + Pulse
# ---------------------------------------------------------------------------


async def execute_heartbeat() -> None:
    """Run one heartbeat cycle using the new Worker + Pulse architecture.

    Runs the worker first (heavy tasks), then the pulse (check-in report).
    This function MUST NOT raise exceptions — all errors are caught and reported.
    """
    try:
        from agents.seo_agent.worker import execute_worker_cycle

        logger.info("Heartbeat starting (worker + pulse mode)...")
        worker_result = await execute_worker_cycle()
        logger.info("Worker completed: %s", worker_result.get("status", "unknown"))

    except Exception:
        logger.error("Worker phase crashed (contained): %s", traceback.format_exc())
        try:
            await send_telegram(f"Worker crashed: {traceback.format_exc()[-300:]}")
        except Exception:
            pass

    try:
        from agents.seo_agent.pulse import execute_pulse

        pulse_result = await execute_pulse()
        logger.info("Pulse completed: %s", pulse_result.get("status", "unknown"))

    except Exception:
        logger.error("Pulse phase crashed (contained): %s", traceback.format_exc())
        try:
            await send_telegram(f"Pulse crashed: {traceback.format_exc()[-200:]}")
        except Exception:
            pass


async def execute_worker_only() -> None:
    """Run only the worker (heavy background tasks)."""
    try:
        from agents.seo_agent.worker import execute_worker_cycle

        await execute_worker_cycle()
    except Exception:
        logger.error("Worker crashed: %s", traceback.format_exc())


async def execute_pulse_only() -> None:
    """Run only the pulse (lightweight check-in)."""
    try:
        from agents.seo_agent.pulse import execute_pulse

        await execute_pulse()
    except Exception:
        logger.error("Pulse crashed: %s", traceback.format_exc())


# ---------------------------------------------------------------------------
# Legacy monolithic heartbeat (kept for backward compatibility)
# ---------------------------------------------------------------------------


async def execute_heartbeat_legacy() -> None:
    """Run the original monolithic heartbeat (pre-OpenClaw architecture).

    Kept for backward compatibility and as a fallback. Use ``--legacy`` flag.
    """
    try:
        await _execute_heartbeat_legacy_inner()
    except Exception:
        logger.error("Legacy heartbeat crashed (contained): %s", traceback.format_exc())
        try:
            await send_telegram(f"Heartbeat crashed: {traceback.format_exc()[-300:]}")
        except Exception:
            pass


async def _execute_heartbeat_legacy_inner() -> None:
    """Original heartbeat logic — preserved as-is for backward compatibility."""
    from agents.seo_agent.config import MAX_WEEKLY_SPEND_USD, SITE_PROFILES
    from agents.seo_agent.tools.crm_tools import get_dashboard_summary
    from agents.seo_agent.tools.supabase_tools import get_weekly_spend

    logger.info("Legacy heartbeat starting...")

    active_sites = {k: v for k, v in SITE_PROFILES.items() if v.get("status") == "active"}

    spend = get_weekly_spend()
    cap = float(os.getenv("MAX_WEEKLY_SPEND_USD", str(MAX_WEEKLY_SPEND_USD)))
    if spend >= cap * 0.95:
        await send_telegram(
            f"Budget alert: ${spend:.2f} / ${cap:.2f} spent this week. "
            f"Pausing autonomous work until next week or you increase the cap."
        )
        return

    try:
        dash = get_dashboard_summary()
    except Exception as e:
        logger.error("Dashboard failed: %s", e)
        return

    kw_count = dash["keywords_discovered"]
    content_count = dash["content_pieces"]

    try:
        from agents.seo_agent.tools.github_tools import list_blog_posts

        actual_post_count = 0
        for site_key in active_sites:
            posts = list_blog_posts(site_key)
            actual_post_count += len(posts)
        content_count = max(content_count, actual_post_count)
    except Exception:
        pass

    prospects_count = dash["prospects_total"]
    gaps_count = dash["content_gaps"]
    report_lines: list[str] = []
    task_executed = False

    try:
        if kw_count == 0:
            await send_telegram("Starting keyword research for all sites...")
            for site in active_sites:
                try:
                    result = run_task("keyword_research", target_site=site)
                    opps = result.get("keyword_opportunities", [])
                    report_lines.append(f"{site}: {len(opps)} keywords found")
                except Exception as e:
                    report_lines.append(f"{site}: keyword research failed — {str(e)[:100]}")
            task_executed = True

        elif gaps_count == 0 and kw_count > 0:
            await send_telegram("Running content gap analysis...")
            for site in ["freeroomplanner", "kitchensdirectory"]:
                try:
                    result = run_task("content_gap", target_site=site)
                    gaps = result.get("content_gaps", [])
                    report_lines.append(f"{site}: {len(gaps)} content gaps found")
                except Exception as e:
                    report_lines.append(f"{site}: gap analysis failed — {str(e)[:100]}")
            task_executed = True

        elif prospects_count == 0:
            await send_telegram("Starting backlink prospecting for freeroomplanner...")
            try:
                result = run_task("discover_prospects", target_site="freeroomplanner")
                prospects = result.get("backlink_prospects", [])
                report_lines.append(f"Found {len(prospects)} backlink prospects")
            except Exception as e:
                report_lines.append(f"Prospecting failed: {str(e)[:150]}")
            task_executed = True

        elif content_count < 30:
            from agents.seo_agent.tools.supabase_tools import query_table

            blog_sites = [s for s in active_sites if s != "ralf_seo" and active_sites[s].get("seed_keywords")]
            target_site_for_post = "freeroomplanner"

            if len(blog_sites) >= 2:
                from agents.seo_agent.tools.github_tools import list_blog_posts as _list_posts

                latest_per_site: dict[str, int] = {}
                for _bs in blog_sites:
                    try:
                        _posts = _list_posts(_bs)
                        latest_per_site[_bs] = len(_posts)
                    except Exception:
                        latest_per_site[_bs] = 0
                target_site_for_post = min(latest_per_site, key=latest_per_site.get)
            elif blog_sites:
                target_site_for_post = blog_sites[0]

            _KEYWORD_BLOCKLIST = {
                "b&q", "bq", "b&q kitchen", "bq kitchen", "b&q kitchen units",
                "bq kitchen units", "b and q", "bandq",
            }

            def _is_blocked(kw_text: str) -> bool:
                kw_lower = kw_text.lower().strip()
                return any(blocked in kw_lower for blocked in _KEYWORD_BLOCKLIST)

            existing_briefs = query_table("seo_content_briefs", limit=500)
            existing_topics = {b.get("keyword", "").lower() for b in existing_briefs}

            existing_slugs: set[str] = set()
            try:
                from agents.seo_agent.tools.github_tools import list_blog_posts, slugify

                existing_posts = list_blog_posts(target_site_for_post)
                existing_slugs = {
                    p.get("name", "").replace(".html", "").replace(".mdx", "").replace(".ts", "").lower()
                    for p in existing_posts
                }
            except Exception:
                pass

            keywords = query_table(
                "seo_keyword_opportunities",
                filters={"target_site": target_site_for_post},
                limit=50,
                order_by="volume",
                order_desc=True,
            )

            untargeted = []
            for k in keywords:
                kw_text = k.get("keyword", "").lower()
                if kw_text in existing_topics:
                    continue
                if _is_blocked(kw_text):
                    continue
                try:
                    slug = slugify(kw_text)
                except Exception:
                    slug = kw_text.replace(" ", "-")
                if slug in existing_slugs:
                    continue
                untargeted.append(k)

            _STOP_WORDS = {
                "uk", "free", "online", "best", "how", "to", "a", "the",
                "for", "in", "of", "your", "and", "with", "guide", "ideas", "tips",
            }

            def _significant_words(text: str) -> set:
                return {w for w in text.lower().split() if w not in _STOP_WORDS and len(w) > 2}

            recent_slugs: list[str] = []
            try:
                from agents.seo_agent.tools.github_tools import list_blog_posts

                recent_posts = list_blog_posts(target_site_for_post)
                recent_slugs = [p.get("name", "").replace(".html", "") for p in recent_posts[:5]]
            except Exception:
                pass

            recent_topics = [_significant_words(slug.replace("-", " ")) for slug in recent_slugs[:3]]

            diverse_keywords: list[dict] = []
            for _kw in untargeted:
                kw_words = _significant_words(_kw.get("keyword", ""))
                too_similar = any(
                    recent and kw_words and len(kw_words & recent) / max(len(kw_words), 1) > 0.5
                    for recent in recent_topics
                )
                if not too_similar:
                    diverse_keywords.append(_kw)

            selected = diverse_keywords if diverse_keywords else untargeted

            if selected:
                kw = selected[0]
                site = target_site_for_post
                kw_text = kw.get("keyword", "")
                await send_telegram(f'Writing new post: "{kw_text}" for {site}...')

                try:
                    from agents.seo_agent.telegram_bot import _generate_blog_post
                    from agents.seo_agent.tools.github_tools import publish_blog_post

                    blog = _generate_blog_post(site, kw_text, kw_text)
                    result = publish_blog_post(
                        site=site,
                        title=blog["title"],
                        content=blog["content"],
                        meta_description=blog["meta_description"],
                    )
                    report_lines.append(f"Published: {blog['title']}\nURL: {result.get('published_url', 'N/A')}")
                except Exception as e:
                    report_lines.append(f"Publishing failed: {str(e)[:150]}")
                task_executed = True
            else:
                report_lines.append("All discovered keywords have content. Running fresh keyword research.")
                for site in blog_sites:
                    try:
                        result = run_task("keyword_research", target_site=site)
                        opps = result.get("keyword_opportunities", [])
                        report_lines.append(f"{site}: {len(opps)} new keywords")
                    except Exception:
                        pass
                task_executed = True

        else:
            from agents.seo_agent.tools.ahrefs_tools import get_organic_keywords
            from agents.seo_agent.tools.crm_tools import snapshot_our_rankings

            for site_key, profile in active_sites.items():
                domain = profile.get("domain", "")
                if domain:
                    try:
                        rankings = get_organic_keywords.invoke(domain)
                        saved = snapshot_our_rankings(site_key, rankings)
                        report_lines.append(f"{site_key}: {saved} rankings tracked")
                    except Exception:
                        report_lines.append(f"{site_key}: ranking snapshot failed")
            task_executed = True

    except Exception as e:
        logger.error("Legacy heartbeat task failed: %s", traceback.format_exc())
        report_lines.append(f"Error: {str(e)[:200]}")

    # Always-run sections (prospecting, CRM promotion, journal, rankings)
    if prospects_count == 0 and kw_count > 5:
        try:
            await send_telegram("Also starting backlink prospecting — our pipeline is empty...")
            result = run_task("discover_prospects", target_site="freeroomplanner")
            prospects = result.get("backlink_prospects", [])
            report_lines.append(f"Prospecting: found {len(prospects)} backlink prospects")
        except Exception as e:
            report_lines.append(f"Prospecting failed: {str(e)[:100]}")

    if prospects_count > 0:
        try:
            from agents.seo_agent.tools.supabase_tools import query_table

            new_prospects = query_table("seo_backlink_prospects", filters={"status": "new"}, limit=20)
            if new_prospects:
                await send_telegram(f"Enriching {len(new_prospects)} new prospects...")
                try:
                    result = run_task("score_prospects", target_site="all")
                    scored = result.get("scored_prospects", [])
                    tier1 = sum(1 for p in scored if p.get("tier") == "tier1")
                    tier2 = sum(1 for p in scored if p.get("tier") == "tier2")
                    report_lines.append(f"Scored {len(scored)} prospects: {tier1} tier-1, {tier2} tier-2")
                except Exception as e:
                    report_lines.append(f"Enrichment failed: {str(e)[:100]}")
        except Exception:
            pass

    try:
        from agents.seo_agent.tools.crm_tools import add_crm_contact, get_crm_contacts
        from agents.seo_agent.tools.supabase_tools import query_table

        scored_prospects = query_table("seo_backlink_prospects", filters={"status": "scored"}, limit=20)
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
        if promoted:
            report_lines.append(f"Added {promoted} prospects to CRM")
    except Exception:
        pass

    try:
        from agents.seo_agent.tools.github_tools import list_blog_posts as _lbp_ralf

        ralf_posts = _lbp_ralf("ralf_seo")
        should_write_journal = len(ralf_posts) < 3
        if not should_write_journal and ralf_posts:
            post_count = len(ralf_posts)
            days_active = max(1, (datetime.now(timezone.utc) - datetime(2026, 3, 26, tzinfo=timezone.utc)).days)
            expected_posts = days_active // 3
            should_write_journal = post_count < expected_posts + 3

        if should_write_journal:
            from agents.seo_agent.tools.reflection_engine import generate_reflective_post

            await send_telegram("Writing a journal entry for ralfseo.com...")
            post = generate_reflective_post()
            from agents.seo_agent.tools.github_tools import publish_blog_post as _pbp_ralf

            result = _pbp_ralf(
                site="ralf_seo",
                title=post["title"],
                content=post["content"],
                meta_description=post["meta_description"],
                category=post.get("category", "Field Report"),
                what_i_learned=post.get("what_i_learned", []),
            )
            report_lines.append(f"Journal: {post['title']}\nURL: {result.get('published_url', 'N/A')}")
    except Exception as e:
        logger.warning("Ralf journal failed: %s", e, exc_info=True)

    try:
        from agents.seo_agent.tools.crm_tools import get_ranking_movers

        for site_key in active_sites:
            movers = get_ranking_movers(site_key, limit=5)
            winners = movers.get("winners", [])
            for w in winners[:3]:
                if (w.get("change") or 0) >= 3:
                    report_lines.append(
                        f"'{w['keyword']}' climbed {w['change']} spots "
                        f"(#{w.get('previous_position')} → #{w.get('position')})"
                    )
    except Exception:
        pass

    if report_lines:
        report = "Here's what happened:\n\n" + "\n".join(report_lines)
        report += f"\n\nBudget: ${spend:.2f} of ${cap:.2f} used this week."
        await send_telegram(report)
    elif not task_executed:
        logger.info("Nothing to do this cycle.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the heartbeat with mode selection."""
    import argparse

    parser = argparse.ArgumentParser(description="Ralf heartbeat orchestrator")
    parser.add_argument("--loop", action="store_true", help="Run on internal schedule")
    parser.add_argument("--interval", type=int, default=180, help="Minutes between runs (default 3 hours)")
    parser.add_argument("--legacy", action="store_true", help="Use legacy monolithic heartbeat")
    parser.add_argument("--worker", action="store_true", help="Run worker only (heavy tasks)")
    parser.add_argument("--pulse", action="store_true", help="Run pulse only (check-in)")
    args = parser.parse_args()

    # Select execution function
    if args.legacy:
        run_fn = execute_heartbeat_legacy
        logger.info("Running in legacy mode")
    elif args.worker:
        run_fn = execute_worker_only
        logger.info("Running worker only")
    elif args.pulse:
        run_fn = execute_pulse_only
        logger.info("Running pulse only")
    else:
        run_fn = execute_heartbeat
        logger.info("Running worker + pulse")

    if args.loop:
        import time

        logger.info("Heartbeat loop starting (every %d minutes)...", args.interval)
        while True:
            try:
                asyncio.run(run_fn())
            except Exception:
                logger.error("Heartbeat cycle failed: %s", traceback.format_exc())
            logger.info("Sleeping %d minutes until next heartbeat...", args.interval)
            time.sleep(args.interval * 60)
    else:
        asyncio.run(run_fn())


if __name__ == "__main__":
    main()
