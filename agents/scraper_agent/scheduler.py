"""Scheduler — runs scraping batches at regular intervals."""

import asyncio
import logging
import traceback
from datetime import datetime, timezone

from agents.scraper_agent.config import DAILY_TARGET, SCRAPE_INTERVAL_HOURS

logger = logging.getLogger(__name__)

# Track daily progress
_today_count = 0
_today_date = ""


async def run_scheduled_batch(send_message) -> None:
    """Run a single scheduled batch. Called by the Telegram bot's job scheduler."""
    global _today_count, _today_date

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if _today_date != today:
        _today_count = 0
        _today_date = today

    remaining = DAILY_TARGET - _today_count
    if remaining <= 0:
        logger.info("Daily target of %d already reached. Skipping batch.", DAILY_TARGET)
        return

    # Run 4 batches per day (every 6 hours) — divide target across batches
    batches_per_day = max(1, 24 // SCRAPE_INTERVAL_HOURS)
    batch_target = max(5, remaining // max(1, batches_per_day - (_today_count > 0)))

    await send_message(f"Starting scrape batch — target: {batch_target} companies (today: {_today_count}/{DAILY_TARGET})")

    try:
        from agents.scraper_agent.tools.firecrawl_client import run_daily_batch
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: run_daily_batch(daily_target=batch_target)
        )

        added = result.get("added", 0)
        _today_count += added

        lines = [
            f"Batch complete: {added} companies added",
            f"Today's progress: {_today_count}/{DAILY_TARGET}",
        ]
        by_seg = result.get("by_segment", {})
        if by_seg:
            lines.append("Breakdown:")
            for k, v in by_seg.items():
                lines.append(f"  {k}: {v} found")

        await send_message("\n".join(lines))

    except Exception as e:
        logger.error("Scheduled batch failed: %s", traceback.format_exc())
        await send_message(f"Batch failed: {str(e)[:300]}")
