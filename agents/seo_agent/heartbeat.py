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
    """Run one heartbeat cycle: check state, pick highest priority task, execute, report."""
    from agents.seo_agent.tools.crm_tools import get_dashboard_summary
    from agents.seo_agent.strategy import generate_next_steps
    from agents.seo_agent.config import SITE_PROFILES, MAX_WEEKLY_SPEND_USD
    from agents.seo_agent.tools.supabase_tools import get_weekly_spend

    logger.info("Heartbeat starting...")

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
            for site in SITE_PROFILES:
                try:
                    result = run_task("keyword_research", target_site=site)
                    opps = result.get("keyword_opportunities", [])
                    report_lines.append(f"{site}: {len(opps)} keywords found")
                except Exception as e:
                    report_lines.append(f"{site}: keyword research failed — {str(e)[:100]}")
            task_executed = True

        # Priority 2: Keywords but no content → create content briefs + write posts
        elif content_count < 5 and kw_count > 0:
            # Pick the site with the most keywords
            site_kw = {s: d["keywords"] for s, d in dash["sites"].items()}
            best_site = max(site_kw, key=site_kw.get) if any(site_kw.values()) else "freeroomplanner"

            await send_telegram(f"Writing a blog post for {best_site} targeting our best keyword...")

            try:
                from agents.seo_agent.tools.supabase_tools import query_table
                keywords = query_table(
                    "seo_keyword_opportunities",
                    filters={"target_site": best_site},
                    limit=5,
                    order_by="volume",
                    order_desc=True,
                )
                if keywords:
                    top_kw = keywords[0]
                    kw_text = top_kw.get("keyword", "room planner guide")

                    # Generate and publish a blog post
                    from agents.seo_agent.telegram_bot import _generate_blog_post
                    from agents.seo_agent.tools.github_tools import publish_blog_post

                    blog = _generate_blog_post(best_site, kw_text, kw_text)
                    result = publish_blog_post(
                        site=best_site,
                        title=blog["title"],
                        content=blog["content"],
                        meta_description=blog["meta_description"],
                    )
                    report_lines.append(
                        f"Published: {blog['title']}\n"
                        f"URL: {result.get('published_url', 'N/A')}"
                    )
                else:
                    report_lines.append(f"No keywords found for {best_site} to write about.")
            except Exception as e:
                report_lines.append(f"Content creation failed: {str(e)[:150]}")
            task_executed = True

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
            # Find keywords we haven't written about yet
            existing_briefs = query_table("seo_content_briefs", limit=500)
            existing_topics = {b.get("keyword", "").lower() for b in existing_briefs}

            keywords = query_table("seo_keyword_opportunities", limit=50, order_by="volume", order_desc=True)
            untargeted = [k for k in keywords if k.get("keyword", "").lower() not in existing_topics]

            if untargeted:
                kw = untargeted[0]
                site = kw.get("target_site", "freeroomplanner")
                kw_text = kw.get("keyword", "")

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
                # Refresh keywords
                for site in SITE_PROFILES:
                    try:
                        result = run_task("keyword_research", target_site=site)
                        opps = result.get("keyword_opportunities", [])
                        report_lines.append(f"{site}: {len(opps)} new keywords")
                    except Exception:
                        pass
                task_executed = True

        # Priority 6: Track rankings (do this periodically regardless)
        else:
            from agents.seo_agent.tools.ahrefs_tools import get_organic_keywords
            from agents.seo_agent.tools.crm_tools import snapshot_our_rankings

            for site_key, profile in SITE_PROFILES.items():
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
