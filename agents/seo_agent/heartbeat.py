"""Ralf's heartbeat — autonomous scheduled execution.

Runs on a schedule (via Railway cron or APScheduler) and executes the
highest-priority task from the strategy engine. Reports progress and
escalates blockers to the user via Telegram.

Usage:
    python -m agents.seo_agent.heartbeat          # Run once (for cron)
    python -m agents.seo_agent.heartbeat --loop    # Run on internal schedule
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
    """Run an SEO agent task synchronously."""
    from agents.seo_agent.agent import build_graph, create_initial_state
    from agents.seo_agent.tools.supabase_tools import ensure_tables, get_weekly_spend

    ensure_tables()
    weekly_spend = get_weekly_spend()

    state = create_initial_state(task_type=task_type, **kwargs)
    state["llm_spend_this_week"] = weekly_spend

    graph = build_graph()
    return graph.invoke(state)


async def execute_heartbeat() -> None:
    """Run one heartbeat cycle: check state, pick highest priority task, execute, report.
    
    This function MUST NOT raise exceptions — all errors are caught and reported.
    """
    try:
        await _execute_heartbeat_inner()
    except Exception:
        logger.error("Heartbeat crashed (contained): %s", traceback.format_exc())
        try:
            await send_telegram(f"Heartbeat crashed: {traceback.format_exc()[-300:]}")
        except Exception:
            pass


async def _execute_heartbeat_inner() -> None:
    """Inner heartbeat logic — may raise exceptions (caught by execute_heartbeat)."""
    from agents.seo_agent.tools.crm_tools import get_dashboard_summary
    from agents.seo_agent.strategy import generate_next_steps
    from agents.seo_agent.config import SITE_PROFILES, MAX_WEEKLY_SPEND_USD
    from agents.seo_agent.tools.supabase_tools import get_weekly_spend

    logger.info("Heartbeat starting...")

    # Only work on active sites
    active_sites = {k: v for k, v in SITE_PROFILES.items() if v.get("status") == "active"}

    # Check budget
    spend = get_weekly_spend()
    cap = float(os.getenv("MAX_WEEKLY_SPEND_USD", str(MAX_WEEKLY_SPEND_USD)))
    if spend >= cap * 0.95:
        await send_telegram(
            f"Budget alert: ${spend:.2f} / ${cap:.2f} spent this week. "
            f"Pausing autonomous work until next week or you increase the cap."
        )
        logger.info("Budget exhausted, skipping heartbeat")
        return

    # Get current state
    try:
        dash = get_dashboard_summary()
    except Exception as e:
        logger.error("Dashboard failed: %s", e)
        return

    kw_count = dash["keywords_discovered"]
    content_count = dash["content_pieces"]
    prospects_count = dash["prospects_total"]
    gaps_count = dash["content_gaps"]

    logger.info(
        "State: %d keywords, %d content, %d gaps, %d prospects",
        kw_count, content_count, gaps_count, prospects_count,
    )

    # Decision tree — what to do next
    task_executed = False
    report_lines = []

    try:
        # Priority 1: No keywords → run keyword research
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

        # Priority 2: Keywords but no content → falls through to Priority 5
        # (Removed: this was a duplicate blog-writing path without blocklist/diversity checks.
        #  Priority 5 handles all blog writing with proper guards.)

        # Priority 3: No content gaps → run gap analysis
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

        # Priority 4: No prospects → start prospecting
        elif prospects_count == 0:
            await send_telegram("Starting backlink prospecting for freeroomplanner...")
            try:
                result = run_task("discover_prospects", target_site="freeroomplanner")
                prospects = result.get("backlink_prospects", [])
                report_lines.append(f"Found {len(prospects)} backlink prospects")
            except Exception as e:
                report_lines.append(f"Prospecting failed: {str(e)[:150]}")
            task_executed = True

        # Priority 5: Content exists, write more
        elif content_count < 30:
            from agents.seo_agent.tools.supabase_tools import query_table

            # --- Site rotation: alternate between freeroomplanner and kitchen_estimator ---
            # ralf_seo uses reflection engine, not keyword-targeted posts — exclude it here
            blog_sites = [s for s in active_sites if s != "ralf_seo" and active_sites[s].get("seed_keywords")]
            target_site_for_post = "freeroomplanner"  # default fallback

            if len(blog_sites) >= 2:
                from agents.seo_agent.tools.github_tools import list_blog_posts as _list_posts
                latest_per_site: dict[str, int] = {}
                for _bs in blog_sites:
                    try:
                        _posts = _list_posts(_bs)
                        latest_per_site[_bs] = len(_posts)
                    except Exception:
                        latest_per_site[_bs] = 0

                # Write for the site with FEWER posts (simple round-robin)
                target_site_for_post = min(latest_per_site, key=latest_per_site.get)
                logger.info(
                    "Site rotation — post counts: %s → writing for %s",
                    latest_per_site,
                    target_site_for_post,
                )
            elif blog_sites:
                target_site_for_post = blog_sites[0]
            # --- End site rotation ---

            # --- Hard keyword blocklist: topics we've exhausted or shouldn't repeat ---
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

            # Find keywords we haven't written about yet
            existing_briefs = query_table("seo_content_briefs", limit=500)
            existing_topics = {b.get("keyword", "").lower() for b in existing_briefs}

            # Also check existing blog file slugs to avoid re-publishing
            existing_slugs: set[str] = set()
            try:
                from agents.seo_agent.tools.github_tools import list_blog_posts, slugify
                existing_posts = list_blog_posts(target_site_for_post)
                existing_slugs = {p.get("name", "").replace(".html", "").replace(".mdx", "").replace(".ts", "").lower() for p in existing_posts}
                logger.info("Existing blog slugs for %s: %d files", target_site_for_post, len(existing_slugs))
            except Exception:
                logger.warning("Could not fetch existing blog slugs", exc_info=True)

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
                # Skip if already in briefs
                if kw_text in existing_topics:
                    continue
                # Skip if blocklisted
                if _is_blocked(kw_text):
                    logger.info("Blocked keyword: %s", kw_text)
                    continue
                # Skip if a blog with this slug already exists
                slug = slugify(kw_text) if 'slugify' in dir() else kw_text.replace(' ', '-')
                if slug in existing_slugs:
                    logger.info("Slug already exists: %s", slug)
                    continue
                untargeted.append(k)

            # --- Topic diversity: skip keywords too similar to recent posts ---
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

            logger.info("Recent slugs for %s: %s", target_site_for_post, recent_slugs)

            recent_topics = [_significant_words(slug.replace("-", " ")) for slug in recent_slugs[:3]]

            diverse_keywords: list[dict] = []
            filtered_out: list[str] = []
            for _kw in untargeted:
                kw_words = _significant_words(_kw.get("keyword", ""))
                too_similar = False
                for recent in recent_topics:
                    if recent and kw_words:
                        overlap = len(kw_words & recent) / max(len(kw_words), 1)
                        if overlap > 0.5:
                            too_similar = True
                            break
                if too_similar:
                    filtered_out.append(_kw.get("keyword", ""))
                else:
                    diverse_keywords.append(_kw)

            if filtered_out:
                logger.info("Diversity filter removed %d keywords: %s", len(filtered_out), filtered_out[:10])

            # Fall back to unfiltered list if everything got filtered out
            selected = diverse_keywords if diverse_keywords else untargeted
            # --- End topic diversity ---

            if selected:
                kw = selected[0]
                site = target_site_for_post
                kw_text = kw.get("keyword", "")

                logger.info("Selected keyword for %s: %s", site, kw_text)
                await send_telegram(f"Writing new post: \"{kw_text}\" for {site}...")

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
                # Refresh keywords for blog sites only (not ralf_seo)
                for site in blog_sites:
                    try:
                        result = run_task("keyword_research", target_site=site)
                        opps = result.get("keyword_opportunities", [])
                        report_lines.append(f"{site}: {len(opps)} new keywords")
                    except Exception:
                        pass
                task_executed = True

        # Priority 6: Ralf's personal blog — write a reflective post every ~3 days
        elif not task_executed:
            try:
                from agents.seo_agent.tools.github_tools import list_blog_posts
                from agents.seo_agent.tools.reflection_engine import generate_reflective_post

                ralf_posts = list_blog_posts("ralf_seo")
                should_write = True

                if ralf_posts:
                    # Check if most recent post is less than 3 days old
                    # Posts are sorted newest first by GitHub API
                    # Simple heuristic: if we have fewer than 20 posts, wait 3 days between them
                    # As the blog grows, can be adjusted
                    should_write = len(ralf_posts) == 0 or True  # For now, let the heartbeat decide

                if should_write and len(ralf_posts) < 100:
                    await send_telegram("Writing a journal entry for my personal blog...")
                    post = generate_reflective_post()

                    from agents.seo_agent.tools.github_tools import publish_blog_post
                    result = publish_blog_post(
                        site="ralf_seo",
                        title=post["title"],
                        content=post["content"],
                        meta_description=post["meta_description"],
                        category=post.get("category", "Field Report"),
                        what_i_learned=post.get("what_i_learned", []),
                    )
                    report_lines.append(
                        f"Journal entry: {post['title']}\n"
                        f"URL: {result.get('published_url', 'N/A')}"
                    )
                    task_executed = True
            except Exception as e:
                logger.warning("Ralf blog post failed: %s", e, exc_info=True)

        # Priority 7: Track rankings (do this periodically regardless)
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
                    except Exception as e:
                        report_lines.append(f"{site_key}: ranking snapshot failed")
            task_executed = True

    except Exception as e:
        logger.error("Heartbeat task failed: %s", traceback.format_exc())
        report_lines.append(f"Error: {str(e)[:200]}")

    # Always check ranking changes at the end of each heartbeat
    try:
        from agents.seo_agent.tools.crm_tools import get_ranking_movers
        for site_key in active_sites:
            movers = get_ranking_movers(site_key, limit=5)
            winners = movers.get("winners", [])
            if winners:
                for w in winners[:3]:
                    if (w.get("change") or 0) >= 3:  # Only report significant moves
                        report_lines.append(
                            f"📈 {w['keyword']}: position {w.get('previous_position')} → {w.get('position')} (+{w['change']})"
                        )
    except Exception:
        pass

    # Send progress report
    if report_lines:
        report = "Heartbeat update:\n\n" + "\n".join(report_lines)
        report += f"\n\nSpend: ${spend:.4f} / ${cap:.2f}"
        await send_telegram(report)
    elif not task_executed:
        logger.info("Nothing to do this cycle.")


def main() -> None:
    """Run the heartbeat."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true", help="Run on internal schedule")
    parser.add_argument("--interval", type=int, default=360, help="Minutes between runs (default 6 hours)")
    args = parser.parse_args()

    if args.loop:
        import time
        logger.info("Heartbeat loop starting (every %d minutes)...", args.interval)
        while True:
            try:
                asyncio.run(execute_heartbeat())
            except Exception:
                logger.error("Heartbeat cycle failed: %s", traceback.format_exc())
            logger.info("Sleeping %d minutes until next heartbeat...", args.interval)
            time.sleep(args.interval * 60)
    else:
        asyncio.run(execute_heartbeat())


if __name__ == "__main__":
    main()
