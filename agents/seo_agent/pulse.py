"""Pulse — lightweight check-in that always sends a Telegram update.

The pulse runs more frequently than the worker and always produces a user-facing
message. It handles:

1. **Ranking movers**: Notable position changes across all sites.
2. **Budget alerts**: Spend approaching the weekly cap.
3. **Blocker escalation**: Services down, repeated failures, rate limits.
4. **Progress summary**: What the worker did since last pulse.
5. **Memory-driven insights**: Patterns detected from episodic memory.

The pulse never executes heavy tasks — it only reads state and reports.

Usage::

    python -m agents.seo_agent.pulse           # Run once
    python -m agents.seo_agent.pulse --loop     # Run on internal schedule
"""

from __future__ import annotations

import logging
import os
import traceback
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


async def execute_pulse() -> dict[str, Any]:
    """Run one pulse cycle: check state, build report, send to Telegram.

    This function MUST NOT raise exceptions.

    Returns:
        Dict with pulse results.
    """
    try:
        return await _execute_pulse_inner()
    except Exception:
        logger.error("Pulse crashed (contained): %s", traceback.format_exc())
        try:
            from agents.seo_agent.heartbeat import send_telegram

            await send_telegram(f"Pulse error: {traceback.format_exc()[-200:]}")
        except Exception:
            pass
        return {"status": "crashed"}


async def _execute_pulse_inner() -> dict[str, Any]:
    """Inner pulse logic."""
    from agents.seo_agent.config import MAX_WEEKLY_SPEND_USD
    from agents.seo_agent.gateway import Gateway
    from agents.seo_agent.heartbeat import send_telegram
    from agents.seo_agent.memory import Memory
    from agents.seo_agent.wal import WAL

    logger.info("Pulse starting...")

    gw = Gateway()
    gw.boot()
    ctx = gw.get_execution_context()
    memory = Memory()

    sections: list[str] = []
    alerts: list[str] = []

    # --- 1. Budget check ---
    spend_pct = 1.0 - ctx.budget_remaining
    cap = float(os.getenv("MAX_WEEKLY_SPEND_USD", str(MAX_WEEKLY_SPEND_USD)))
    spend_usd = spend_pct * cap

    if ctx.budget_remaining < 0.05:
        alerts.append(
            f"I've almost used the entire weekly budget "
            f"(${spend_usd:.2f} of ${cap:.2f}). "
            f"I'll pause non-essential tasks until the budget resets."
        )
    elif ctx.budget_remaining < 0.2:
        alerts.append(
            f"Heads up — I've used most of the weekly budget "
            f"(${spend_usd:.2f} of ${cap:.2f}). "
            f"Switching to cheaper models to stay within limits."
        )

    # --- 2. Service health ---
    down_services = [
        s.name for s in gw.services.values()
        if s.status == "down"
    ]
    degraded_services = [
        s.name for s in gw.services.values()
        if s.status == "degraded"
    ]
    if down_services:
        alerts.append(
            f"I can't reach {', '.join(down_services)} right now — "
            f"any tasks that depend on {'it' if len(down_services) == 1 else 'them'} "
            f"will be skipped until {'it comes' if len(down_services) == 1 else 'they come'} back."
        )
    if degraded_services:
        svc_names = ", ".join(degraded_services)
        alerts.append(
            f"{svc_names} {'is' if len(degraded_services) == 1 else 'are'} "
            f"responding slowly or intermittently. Some tasks may take longer or fail."
        )

    # --- 3. Rate limits ---
    active_limits = gw.rate_limiter.active_limits()
    if active_limits:
        limited_names = ", ".join(active_limits.keys())
        alerts.append(
            f"I'm being rate-limited by {limited_names}, "
            f"so I'm backing off to avoid getting blocked."
        )

    # --- 4. Ranking movers ---
    try:
        from agents.seo_agent.tools.crm_tools import get_ranking_movers

        for site_key in ctx.active_sites:
            movers = get_ranking_movers(site_key, limit=5)
            winners = movers.get("winners", [])
            losers = movers.get("losers", [])

            for w in winners[:3]:
                change = w.get("change", 0)
                if change >= 3:
                    sections.append(
                        f"'{w['keyword']}' climbed {change} spots "
                        f"(#{w.get('previous_position', '?')} → #{w.get('position', '?')})"
                    )

            for l in losers[:2]:
                change = abs(l.get("change", 0))
                if change >= 5:
                    sections.append(
                        f"'{l['keyword']}' dropped {change} spots "
                        f"(#{l.get('previous_position', '?')} → #{l.get('position', '?')}) — worth keeping an eye on"
                    )
    except Exception:
        logger.warning("Could not fetch ranking movers", exc_info=True)

    # --- 5. Recent worker activity ---
    try:
        wal = WAL()
        recent = wal.get_recent_cycles(limit=3)
        completed_tasks = 0
        failed_tasks = 0
        for cycle in recent:
            for task in cycle.get("tasks_json", []):
                if task.get("status") == "done":
                    completed_tasks += 1
                elif task.get("status") == "failed":
                    failed_tasks += 1

        if completed_tasks or failed_tasks:
            total = completed_tasks + failed_tasks
            if failed_tasks == 0:
                sections.append(
                    f"I completed {completed_tasks} "
                    f"{'task' if completed_tasks == 1 else 'tasks'} recently "
                    f"with no issues."
                )
            elif completed_tasks == 0:
                sections.append(
                    f"Last {total} {'task' if total == 1 else 'tasks'} all failed — "
                    f"something may be wrong."
                )
            else:
                sections.append(
                    f"I ran {total} tasks recently: {completed_tasks} succeeded "
                    f"and {failed_tasks} failed."
                )
    except Exception:
        pass

    # --- 6. Memory-driven insights ---
    try:
        # Check for repeated failures that need escalation
        failure_memories = memory.recall(category="context", limit=10)
        escalation_memories = [
            m for m in failure_memories
            if "escalate" in str(m.get("tags", []))
        ]
        for m in escalation_memories[:2]:
            alerts.append(f"Something I noticed: {m.get('content', '')[:120]}")

        # Surface recent learnings
        learnings = memory.recall(category="learning", limit=3)
        for m in learnings[:1]:
            sections.append(f"I noticed a pattern: {m.get('content', '')[:120]}")
    except Exception:
        pass

    # --- 7. Content & pipeline summary ---
    try:
        from agents.seo_agent.tools.crm_tools import get_dashboard_summary

        dash = get_dashboard_summary()
        kw = dash.get("keywords_discovered", 0)
        posts = dash.get("content_pieces", 0)
        prospects = dash.get("prospects_total", 0)
        sections.append(
            f"Currently tracking {kw} keywords, "
            f"{posts} {'post' if posts == 1 else 'posts'} in the pipeline, "
            f"and {prospects} link-building {'prospect' if prospects == 1 else 'prospects'}."
        )
    except Exception:
        pass

    # Build and send the pulse message
    if not alerts and not sections:
        logger.info("Pulse: nothing to report")
        return {"status": "quiet", "message": "All quiet — nothing new to report."}

    message = _format_pulse(alerts, sections, spend_usd, cap, ctx.budget_remaining)
    await send_telegram(message)

    return {"status": "sent", "alerts": len(alerts), "sections": len(sections)}


def _format_pulse(
    alerts: list[str],
    sections: list[str],
    spend: float,
    cap: float,
    budget_remaining: float,
) -> str:
    """Format the pulse message for Telegram.

    Args:
        alerts: Critical alerts that need attention.
        sections: Informational sections.
        spend: Current spend in USD.
        cap: Weekly budget cap in USD.
        budget_remaining: Budget fraction remaining.

    Returns:
        Formatted message string.
    """
    lines: list[str] = []

    if alerts:
        for a in alerts:
            lines.append(f"⚠️ {a}")
        lines.append("")

    if sections:
        for s in sections:
            lines.append(s)
        lines.append("")

    lines.append(f"Budget: ${spend:.2f} of ${cap:.2f} used this week ({budget_remaining:.0%} left).")

    return "\n".join(lines)


def main() -> None:
    """Run the pulse."""
    import argparse
    import asyncio

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Ralf SEO pulse (lightweight check-in)")
    parser.add_argument("--loop", action="store_true", help="Run on internal schedule")
    parser.add_argument("--interval", type=int, default=60, help="Minutes between runs (default 1 hour)")
    args = parser.parse_args()

    if args.loop:
        import time

        logger.info("Pulse loop starting (every %d minutes)...", args.interval)
        while True:
            try:
                asyncio.run(execute_pulse())
            except Exception:
                logger.error("Pulse cycle failed: %s", traceback.format_exc())
            logger.info("Sleeping %d minutes until next pulse...", args.interval)
            time.sleep(args.interval * 60)
    else:
        asyncio.run(execute_pulse())


if __name__ == "__main__":
    main()
