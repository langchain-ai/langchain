"""Scraper agent Telegram bot — standalone data collection interface."""

import asyncio
import logging
import traceback
from functools import partial

from telegram import BotCommand, Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from agents.scraper_agent.config import TELEGRAM_BOT_TOKEN, TELEGRAM_OWNER_CHAT_ID, SCRAPE_INTERVAL_HOURS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Scraper agent online. I find kitchen and bathroom companies using "
        "Tavily search + Firecrawl extraction, and add them to the shared CRM.\n\n"
        "Commands:\n"
        "/scrape <country> [category] — run a scrape (UK, US, CA)\n"
        "/scrape_all — scrape all countries\n"
        "/batch — run a daily batch (targets 50/day)\n"
        "/status — CRM stats\n"
        "/today — today's scraping progress\n"
        "/help — show commands"
    )


async def cmd_scrape(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = context.args or []
    country = args[0].upper() if args else "UK"
    category = args[1] if len(args) > 1 else "kitchen_company"

    if country not in ("UK", "US", "CA"):
        await update.message.reply_text("Country must be UK, US, or CA.")
        return

    await update.message.reply_text(f"Scraping {category} in {country}...")
    try:
        from agents.scraper_agent.tools.firecrawl_client import run_scraper
        result = await asyncio.get_event_loop().run_in_executor(
            None, partial(run_scraper, country=country, category=category, max_queries=10)
        )
        lines = [
            f"Scrape complete: {category} in {country}",
            f"URLs found: {result.get('urls_found', 0)}",
            f"Added to CRM: {result.get('added_to_crm', 0)}",
            f"Skipped: {result.get('skipped', 0)}",
        ]
        await update.message.reply_text("\n".join(lines))
    except Exception as e:
        await update.message.reply_text(f"Failed: {str(e)[:300]}")


async def cmd_scrape_all(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Running full scrape across UK, US, CA...")
    try:
        from agents.scraper_agent.tools.firecrawl_client import run_full_scrape
        result = await asyncio.get_event_loop().run_in_executor(
            None, partial(run_full_scrape, max_queries_per_country=5)
        )
        total = result.get("total_added", 0)
        lines = [f"Full scrape: {total} companies added\n"]
        for key, data in result.items():
            if key == "total_added":
                continue
            if isinstance(data, dict):
                lines.append(f"  {key}: {data.get('added_to_crm', 0)} added")
        await update.message.reply_text("\n".join(lines))
    except Exception as e:
        await update.message.reply_text(f"Failed: {str(e)[:300]}")


async def cmd_batch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Manually trigger a daily batch."""
    from agents.scraper_agent.scheduler import run_scheduled_batch

    async def send_msg(text):
        await update.message.reply_text(text)

    await run_scheduled_batch(send_msg)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        from agents.scraper_agent.tools.crm_client import get_crm_stats
        stats = get_crm_stats()
        lines = [f"CRM: {stats['total']} total contacts\n"]

        if stats["by_country"]:
            lines.append("By country:")
            for k, v in sorted(stats["by_country"].items(), key=lambda x: -x[1]):
                lines.append(f"  {k}: {v}")

        if stats["by_category"]:
            lines.append("\nBy category:")
            for k, v in sorted(stats["by_category"].items(), key=lambda x: -x[1]):
                lines.append(f"  {k}: {v}")

        if stats["by_source"]:
            lines.append("\nBy source:")
            for k, v in sorted(stats["by_source"].items(), key=lambda x: -x[1]):
                lines.append(f"  {k}: {v}")

        await update.message.reply_text("\n".join(lines))
    except Exception as e:
        await update.message.reply_text(f"Status error: {str(e)[:200]}")


async def cmd_today(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from agents.scraper_agent.scheduler import _today_count, _today_date
    from agents.scraper_agent.config import DAILY_TARGET
    await update.message.reply_text(
        f"Today ({_today_date or 'not started'}):\n"
        f"Companies added: {_today_count}/{DAILY_TARGET}"
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await cmd_start(update, context)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle natural language messages — simple keyword matching."""
    text = (update.message.text or "").lower().strip()

    if any(w in text for w in ["status", "stats", "how many", "count"]):
        await cmd_status(update, context)
    elif any(w in text for w in ["scrape", "find", "search", "run"]):
        # Try to extract country
        country = "UK"
        if "us" in text or "america" in text:
            country = "US"
        elif "canada" in text or "ca" in text:
            country = "CA"
        context.args = [country]
        await cmd_scrape(update, context)
    elif any(w in text for w in ["batch", "daily", "target"]):
        await cmd_batch(update, context)
    elif any(w in text for w in ["today", "progress"]):
        await cmd_today(update, context)
    else:
        await update.message.reply_text(
            "I'm the scraper agent — I find companies using Tavily + Firecrawl.\n"
            "Try: /scrape UK, /batch, /status, or /today"
        )


async def _scheduled_batch_callback(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Called by APScheduler to run periodic scraping batches."""
    from agents.scraper_agent.scheduler import run_scheduled_batch

    chat_id = TELEGRAM_OWNER_CHAT_ID
    if not chat_id:
        return

    async def send_msg(text):
        await context.bot.send_message(chat_id=int(chat_id), text=text)

    try:
        await run_scheduled_batch(send_msg)
    except Exception:
        logger.error("Scheduled batch crashed: %s", traceback.format_exc())


def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        logger.error("SCRAPER_TELEGRAM_BOT_TOKEN not set")
        return

    logger.info("Starting scraper agent Telegram bot...")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("scrape", cmd_scrape))
    app.add_handler(CommandHandler("scrape_all", cmd_scrape_all))
    app.add_handler(CommandHandler("batch", cmd_batch))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("today", cmd_today))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Set bot commands
    async def post_init(application):
        await application.bot.set_my_commands([
            BotCommand("scrape", "Scrape companies (country, category)"),
            BotCommand("scrape_all", "Scrape all countries"),
            BotCommand("batch", "Run daily batch"),
            BotCommand("status", "CRM stats"),
            BotCommand("today", "Today's progress"),
            BotCommand("help", "Show commands"),
        ])

        # Schedule periodic batches
        if SCRAPE_INTERVAL_HOURS > 0 and TELEGRAM_OWNER_CHAT_ID:
            application.job_queue.run_repeating(
                _scheduled_batch_callback,
                interval=SCRAPE_INTERVAL_HOURS * 3600,
                first=600,  # First run after 10 minutes
                name="scraper_batch",
            )
            logger.info("Scheduled scraping every %d hours", SCRAPE_INTERVAL_HOURS)

    app.post_init = post_init

    logger.info("Scraper agent bot starting polling...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
