"""Telegram bot interface for the SEO agent.

Runs as a long-polling process — suitable for Railway worker deployments.
Set ``TELEGRAM_BOT_TOKEN`` in the environment to start.

Usage::

    python -m agents.seo_agent.telegram_bot

"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import traceback
from collections import defaultdict, deque
from functools import partial

from dotenv import load_dotenv
from telegram import BotCommand, Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# Ensure repo root is on sys.path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("telegram_bot")

# ---------------------------------------------------------------------------
# Authorisation — only your Telegram user ID can interact
# ---------------------------------------------------------------------------

ALLOWED_USER_IDS: set[int] = set()
_raw = os.getenv("TELEGRAM_ALLOWED_USERS", "")
if _raw:
    ALLOWED_USER_IDS = {int(uid.strip()) for uid in _raw.split(",") if uid.strip()}


def _is_authorised(update: Update) -> bool:
    """Return True if the message sender is authorised (or if no allowlist is set)."""
    if not ALLOWED_USER_IDS:
        return True  # No allowlist = open to anyone
    user = update.effective_user
    return user is not None and user.id in ALLOWED_USER_IDS


async def _check_auth(update: Update) -> bool:
    """Check auth and send rejection message if not authorised."""
    if _is_authorised(update):
        return True
    await update.message.reply_text("⛔ Unauthorised. Your user ID is not in the allowlist.")
    return False


# ---------------------------------------------------------------------------
# Graph runner — executes SEO agent tasks in a thread pool
# ---------------------------------------------------------------------------


def _run_graph_sync(task_type: str, **kwargs) -> dict:
    """Run the SEO agent graph synchronously (called from thread pool)."""
    from agents.seo_agent.agent import build_graph, create_initial_state
    from agents.seo_agent.tools.supabase_tools import ensure_tables, get_weekly_spend

    ensure_tables()
    weekly_spend = get_weekly_spend()

    state = create_initial_state(task_type=task_type, **kwargs)
    state["llm_spend_this_week"] = weekly_spend

    graph = build_graph()
    return graph.invoke(state)


async def _run_agent_task(
    update: Update,
    task_type: str,
    **kwargs,
) -> dict:
    """Run an agent task and handle errors gracefully."""
    await update.message.reply_text(f"⏳ Running `{task_type}`...", parse_mode="Markdown")
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, partial(_run_graph_sync, task_type, **kwargs)
        )
        if result.get("errors"):
            error_text = "\n".join(str(e) for e in result["errors"][:3])
            await update.message.reply_text(f"⚠️ Completed with errors:\n```\n{error_text}\n```", parse_mode="Markdown")
        return result
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Agent task %s failed: %s", task_type, tb)
        await update.message.reply_text(
            f"❌ `{task_type}` failed:\n```\n{str(e)[:500]}\n```",
            parse_mode="Markdown",
        )
        return {}


# ---------------------------------------------------------------------------
# Bot command handlers
# ---------------------------------------------------------------------------


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start — welcome message."""
    if not await _check_auth(update):
        return
    await update.message.reply_text(
        "👋 *Ralf SEO Bot*\n\n"
        "I'm your SEO agent for kitchensdirectory, freeroomplanner, and kitchen estimator.\n\n"
        "*Commands:*\n"
        "/keyword\\_research `<site>` `[seed]` — Find keyword opportunities\n"
        "/content\\_gap `<site>` — Find content gaps vs competitors\n"
        "/content\\_brief `<site>` `<keyword>` — Generate a content brief\n"
        "/discover\\_prospects `<site>` — Find backlink prospects\n"
        "/score\\_prospects — Score all enriched prospects\n"
        "/generate\\_emails — Generate outreach emails\n"
        "/outreach\\_report — Weekly outreach summary\n"
        "/cost\\_report — LLM spend this week\n"
        "/rank\\_report `[site]` — Ranking report\n"
        "/weekly\\_report — Full weekly SEO report\n"
        "/status — Quick system check\n\n"
        "Sites: `kitchensdirectory`, `freeroomplanner`, `kitchen_estimator`",
        parse_mode="Markdown",
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status — quick health check."""
    if not await _check_auth(update):
        return
    from agents.seo_agent.tools.supabase_tools import get_weekly_spend, query_table
    from agents.seo_agent.tools.llm_router import _use_openrouter

    try:
        spend = get_weekly_spend()
        cap = float(os.getenv("MAX_WEEKLY_SPEND_USD", "50.00"))
        prospects = query_table("seo_backlink_prospects", limit=1)
        keywords = query_table("seo_keyword_opportunities", limit=1)

        await update.message.reply_text(
            "✅ *System Status*\n\n"
            f"LLM Provider: `{'OpenRouter' if _use_openrouter() else 'Anthropic'}`\n"
            f"Weekly spend: `${spend:.4f}` / `${cap:.2f}`\n"
            f"Supabase: `connected`\n"
            f"Prospects in DB: `{'yes' if prospects else 'empty'}`\n"
            f"Keywords in DB: `{'yes' if keywords else 'empty'}`",
            parse_mode="Markdown",
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Status check failed:\n```\n{e}\n```", parse_mode="Markdown")


async def cmd_keyword_research(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /keyword_research <site> [seed keyword]."""
    if not await _check_auth(update):
        return
    args = context.args or []
    if not args:
        await update.message.reply_text("Usage: `/keyword_research <site> [seed keyword]`\nExample: `/keyword_research kitchensdirectory bespoke kitchens`", parse_mode="Markdown")
        return

    site = args[0]
    seed = " ".join(args[1:]) if len(args) > 1 else None

    result = await _run_agent_task(update, "keyword_research", target_site=site, seed_keyword=seed)
    opportunities = result.get("keyword_opportunities", [])

    if not opportunities:
        await update.message.reply_text("No keyword opportunities found.")
        return

    lines = [f"🔍 *{len(opportunities)} keyword opportunities:*\n"]
    for kw in opportunities[:15]:
        vol = kw.get("volume", "?")
        kd = kw.get("kd", "?")
        intent = kw.get("intent", "")
        lines.append(f"• `{kw.get('keyword', 'N/A')}` — vol:{vol} KD:{kd} {intent}")

    if len(opportunities) > 15:
        lines.append(f"\n_…and {len(opportunities) - 15} more_")

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_content_gap(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /content_gap <site>."""
    if not await _check_auth(update):
        return
    args = context.args or []
    if not args:
        await update.message.reply_text("Usage: `/content_gap <site>`", parse_mode="Markdown")
        return

    result = await _run_agent_task(update, "content_gap", target_site=args[0])
    gaps = result.get("content_gaps", [])

    if not gaps:
        await update.message.reply_text("No content gaps found.")
        return

    lines = [f"📊 *{len(gaps)} content gaps:*\n"]
    for g in gaps[:15]:
        lines.append(f"• `{g.get('keyword', 'N/A')}` — vol:{g.get('volume', '?')} stage:{g.get('funnel_stage', '?')}")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_content_brief(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /content_brief <site> <keyword>."""
    if not await _check_auth(update):
        return
    args = context.args or []
    if len(args) < 2:
        await update.message.reply_text("Usage: `/content_brief <site> <keyword phrase>`", parse_mode="Markdown")
        return

    site = args[0]
    keyword = " ".join(args[1:])

    result = await _run_agent_task(update, "content_brief", target_site=site, selected_keyword=keyword)
    brief = result.get("content_brief")

    if brief:
        title = brief.get("title", keyword)
        meta = brief.get("meta_description", "")
        wc = brief.get("target_word_count", "?")
        headings = brief.get("headings", [])

        text = f"📝 *Content Brief: {title}*\n\n"
        text += f"Meta: _{meta[:200]}_\n" if meta else ""
        text += f"Target: {wc} words\n\n"
        if headings:
            text += "*Headings:*\n"
            for h in headings[:10]:
                text += f"• {h}\n"
        await update.message.reply_text(text, parse_mode="Markdown")
    else:
        await update.message.reply_text("Failed to generate content brief.")


async def cmd_discover_prospects(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /discover_prospects <site>."""
    if not await _check_auth(update):
        return
    args = context.args or []
    if not args:
        await update.message.reply_text("Usage: `/discover_prospects <site>`", parse_mode="Markdown")
        return

    result = await _run_agent_task(update, "discover_prospects", target_site=args[0])
    prospects = result.get("backlink_prospects", [])

    if not prospects:
        await update.message.reply_text("No prospects found.")
        return

    lines = [f"🔗 *{len(prospects)} backlink prospects:*\n"]
    for p in prospects[:15]:
        dr = p.get("dr", "?")
        lines.append(f"• DR:{dr} `{p.get('domain', 'N/A')}` ({p.get('discovery_method', '?')})")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_score_prospects(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /score_prospects."""
    if not await _check_auth(update):
        return
    result = await _run_agent_task(update, "score_prospects", target_site="all")
    scored = result.get("scored_prospects", [])
    tier1 = sum(1 for p in scored if p.get("tier") == "tier1")
    tier2 = sum(1 for p in scored if p.get("tier") == "tier2")
    rejected = sum(1 for p in scored if p.get("status") == "rejected")

    await update.message.reply_text(
        f"📊 *Scored {len(scored)} prospects*\n\n"
        f"Tier 1: {tier1}\nTier 2: {tier2}\nRejected: {rejected}",
        parse_mode="Markdown",
    )


async def cmd_generate_emails(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /generate_emails."""
    if not await _check_auth(update):
        return
    result = await _run_agent_task(update, "generate_emails", target_site="all")
    emails = result.get("emails_generated", [])

    if not emails:
        await update.message.reply_text("No emails generated (check if prospects are scored).")
        return

    lines = [f"✉️ *{len(emails)} outreach emails generated:*\n"]
    for e in emails[:10]:
        lines.append(f"• Tier {e.get('tier', '?')} → `{e.get('contact_email', 'N/A')}` | {e.get('subject', 'N/A')[:50]}")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_cost_report(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /cost_report."""
    if not await _check_auth(update):
        return
    from agents.seo_agent.tools.supabase_tools import get_weekly_spend, query_table

    spend = get_weekly_spend()
    cap = float(os.getenv("MAX_WEEKLY_SPEND_USD", "50.00"))
    remaining = max(0.0, cap - spend)
    pct = (spend / cap * 100) if cap > 0 else 0

    text = f"💰 *LLM Cost Report*\n\nSpent: `${spend:.4f}`\nBudget: `${cap:.2f}`\nRemaining: `${remaining:.4f}` ({100 - pct:.1f}%)\n"

    if pct >= 80:
        text += "\n⚠️ Budget >80% — models will be downgraded"

    logs = query_table("llm_cost_log", limit=500)
    if logs:
        by_task: dict[str, float] = {}
        for row in logs:
            task = row.get("task_type", "unknown")
            by_task[task] = by_task.get(task, 0.0) + row.get("cost_usd", 0.0)
        text += "\n*Breakdown:*\n"
        for task, cost in sorted(by_task.items(), key=lambda x: -x[1])[:10]:
            text += f"• `{task}`: ${cost:.4f}\n"

    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_rank_report(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /rank_report [site]."""
    if not await _check_auth(update):
        return
    args = context.args or []
    site = args[0] if args else "all"
    result = await _run_agent_task(update, "rank_report", target_site=site)
    data = result.get("rank_data", [])

    if not data:
        await update.message.reply_text("No ranking data available.")
        return

    lines = [f"📈 *Rank report ({len(data)} keywords):*\n"]
    for row in data[:20]:
        pos = row.get("position", "?")
        prev = row.get("previous_position")
        change = ""
        if isinstance(pos, (int, float)) and isinstance(prev, (int, float)):
            diff = prev - pos
            change = f" ({'↑' if diff > 0 else '↓'}{abs(diff):.0f})" if diff != 0 else ""
        lines.append(f"• `{row.get('keyword', 'N/A')[:35]}` → pos {pos}{change}")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_weekly_report(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /weekly_report."""
    if not await _check_auth(update):
        return
    result = await _run_agent_task(update, "weekly_report", target_site="all")
    report = result.get("report", "")
    if report:
        # Telegram has a 4096 char limit per message
        for i in range(0, len(report), 4000):
            await update.message.reply_text(report[i:i + 4000])
    else:
        await update.message.reply_text("No report generated.")


async def cmd_outreach_report(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /outreach_report."""
    if not await _check_auth(update):
        return
    result = await _run_agent_task(update, "outreach_report", target_site="all")
    report = result.get("report", "")
    if report:
        for i in range(0, len(report), 4000):
            await update.message.reply_text(report[i:i + 4000])
    else:
        await update.message.reply_text("No outreach data to report.")


async def cmd_unknown(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle unknown commands — route through natural language handler."""
    if not await _check_auth(update):
        return
    # Strip the slash and treat as natural language
    text = update.message.text
    if text:
        update.message.text = text.lstrip("/").replace("_", " ")
        await handle_natural_language(update, context)


# ---------------------------------------------------------------------------
# Natural language conversation handler
# ---------------------------------------------------------------------------

# Per-user conversation history (last 20 messages)
_conversation_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=20))

_NL_SYSTEM_PROMPT = """You are Ralf, an SEO assistant bot for a portfolio of websites:
- kitchensdirectory.co.uk — a directory of UK kitchen makers
- freeroomplanner.com — a free online room planning tool
- kitchen_estimator — a kitchen renovation cost estimator

You help the user manage SEO for these sites. You can run the following tasks by
returning a JSON action block. If the user's message maps to one of these tasks,
return ONLY a JSON block (no other text) in this exact format:

```json
{"action": "<task_type>", "params": {"target_site": "...", ...}}
```

Available actions and their parameters:
- keyword_research: {target_site, seed_keyword?} — Find keyword opportunities
- content_gap: {target_site} — Find content gaps vs competitors
- content_brief: {target_site, selected_keyword} — Generate a content brief
- write_content: {target_site, brief_id} — Write content from a brief
- discover_prospects: {target_site} — Find backlink prospects
- enrich_prospects: {target_site:"all"} — Enrich prospect data
- score_prospects: {target_site:"all"} — Score all enriched prospects
- generate_emails: {target_site:"all"} — Generate outreach emails
- rank_report: {target_site} — Ranking report
- weekly_report: {target_site:"all"} — Full weekly SEO report
- outreach_report: {target_site:"all"} — Outreach summary
- cost_report: (no params, handled directly)
- status: (no params, handled directly)

Site keys: "kitchensdirectory", "freeroomplanner", "kitchen_estimator", "all"

Rules:
1. If the user clearly wants to run a task, return the JSON action block ONLY.
2. If the user asks about costs/spend, return: {"action": "cost_report"}
3. If the user asks about system status, return: {"action": "status"}
4. If the user's intent is ambiguous, ask a clarifying question in plain text.
5. If the user is just chatting, asking questions about SEO strategy, or asking
   about results you previously shared, respond conversationally in plain text.
6. Keep responses concise — this is Telegram, not email.
7. When the user says a site name casually (e.g. "kitchens directory", "the directory",
   "room planner"), map it to the correct site key.
8. Default to "kitchensdirectory" if the user doesn't specify a site and the context
   suggests kitchens.
9. You DO have memory within this conversation — you can reference earlier messages.
   Results from tasks are also saved to the Supabase database and persist across sessions.
10. Never say you don't have memory or can't remember things. You have conversation
    history and a database backend.
"""


def _call_openrouter_sync(
    messages: list[dict[str, str]],
    system: str,
) -> str:
    """Call OpenRouter synchronously for the NL router (Haiku — fast + cheap)."""
    from agents.seo_agent.tools.llm_router import (
        _get_openrouter_client,
        _get_anthropic_client,
        _use_openrouter,
        _resolve_model_id,
        _HAIKU,
    )

    oai_messages = [{"role": "system", "content": system}] + messages

    if _use_openrouter():
        client = _get_openrouter_client()
        response = client.chat.completions.create(
            model=_resolve_model_id(_HAIKU),
            max_tokens=500,
            messages=oai_messages,
        )
        return response.choices[0].message.content or ""
    else:
        client = _get_anthropic_client()
        response = client.messages.create(
            model=_resolve_model_id(_HAIKU),
            max_tokens=500,
            system=system,
            messages=messages,
        )
        return "".join(b.text for b in response.content if b.type == "text")


def _parse_action(text: str) -> dict | None:
    """Try to extract a JSON action block from the LLM response."""
    text = text.strip()
    # Try direct JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "action" in data:
            return data
    except (json.JSONDecodeError, ValueError):
        pass
    # Try extracting from markdown code block
    if "```" in text:
        for block in text.split("```"):
            block = block.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            try:
                data = json.loads(block)
                if isinstance(data, dict) and "action" in data:
                    return data
            except (json.JSONDecodeError, ValueError):
                continue
    return None


async def _format_task_result(action: str, result: dict) -> str:
    """Format agent task results into a readable Telegram message."""
    if not result:
        return "The task didn't return any results."

    if result.get("errors"):
        error_text = "\n".join(str(e) for e in result["errors"][:3])
        return f"⚠️ Completed with errors:\n{error_text}"

    parts = []

    if action == "keyword_research":
        opps = result.get("keyword_opportunities", [])
        if opps:
            parts.append(f"Found {len(opps)} keyword opportunities:\n")
            for kw in opps[:12]:
                parts.append(f"• {kw.get('keyword', 'N/A')} — vol:{kw.get('volume', '?')} KD:{kw.get('kd', '?')}")
            if len(opps) > 12:
                parts.append(f"\n…and {len(opps) - 12} more saved to the database.")
        else:
            parts.append("No keyword opportunities found.")

    elif action == "content_gap":
        gaps = result.get("content_gaps", [])
        if gaps:
            parts.append(f"Found {len(gaps)} content gaps:\n")
            for g in gaps[:12]:
                parts.append(f"• {g.get('keyword', 'N/A')} — vol:{g.get('volume', '?')} stage:{g.get('funnel_stage', '?')}")
        else:
            parts.append("No content gaps found.")

    elif action == "content_brief":
        brief = result.get("content_brief")
        if brief:
            parts.append(f"Content Brief: {brief.get('title', 'Untitled')}\n")
            if brief.get("meta_description"):
                parts.append(f"Meta: {brief['meta_description'][:200]}\n")
            parts.append(f"Target: {brief.get('target_word_count', '?')} words")
            headings = brief.get("headings", [])
            if headings:
                parts.append("\nHeadings:")
                for h in headings[:10]:
                    parts.append(f"• {h}")
        else:
            parts.append("Failed to generate content brief.")

    elif action == "discover_prospects":
        prospects = result.get("backlink_prospects", [])
        if prospects:
            parts.append(f"Found {len(prospects)} backlink prospects:\n")
            for p in prospects[:12]:
                parts.append(f"• DR:{p.get('dr', '?')} {p.get('domain', 'N/A')} ({p.get('discovery_method', '?')})")
        else:
            parts.append("No prospects found.")

    elif action == "score_prospects":
        scored = result.get("scored_prospects", [])
        tier1 = sum(1 for p in scored if p.get("tier") == "tier1")
        tier2 = sum(1 for p in scored if p.get("tier") == "tier2")
        rejected = sum(1 for p in scored if p.get("status") == "rejected")
        parts.append(f"Scored {len(scored)} prospects:\nTier 1: {tier1}\nTier 2: {tier2}\nRejected: {rejected}")

    elif action == "generate_emails":
        emails = result.get("emails_generated", [])
        if emails:
            parts.append(f"Generated {len(emails)} outreach emails:\n")
            for e in emails[:8]:
                parts.append(f"• Tier {e.get('tier', '?')} → {e.get('contact_email', 'N/A')} | {e.get('subject', '')[:50]}")
        else:
            parts.append("No emails generated — check if prospects are scored.")

    elif action in ("rank_report",):
        data = result.get("rank_data", [])
        if data:
            parts.append(f"Rank report ({len(data)} keywords):\n")
            for row in data[:15]:
                pos = row.get("position", "?")
                prev = row.get("previous_position")
                change = ""
                if isinstance(pos, (int, float)) and isinstance(prev, (int, float)):
                    diff = prev - pos
                    if diff != 0:
                        change = f" ({'↑' if diff > 0 else '↓'}{abs(diff):.0f})"
                parts.append(f"• {row.get('keyword', 'N/A')[:35]} → pos {pos}{change}")
        else:
            parts.append("No ranking data available.")

    elif action in ("weekly_report", "outreach_report"):
        report = result.get("report", "")
        parts.append(report if report else "No report data available.")

    else:
        # Generic fallback
        parts.append(f"Task `{action}` completed.")
        for key, val in result.items():
            if key not in ("errors", "llm_spend_this_week", "task_type") and val:
                if isinstance(val, list):
                    parts.append(f"{key}: {len(val)} items")
                elif isinstance(val, str) and len(val) > 10:
                    parts.append(f"{key}: {val[:200]}")

    return "\n".join(parts) if parts else "Task completed with no output."


async def handle_natural_language(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle free-text messages by routing through Claude."""
    try:
        if not await _check_auth(update):
            return

        user_id = update.effective_user.id
        user_text = update.message.text or ""
        if not user_text.strip():
            return

        logger.info("NL message from %s: %s", user_id, user_text[:100])

        # Add to conversation history
        history = _conversation_history[user_id]
        history.append({"role": "user", "content": user_text})

        # Build messages for the LLM (include history for context)
        messages = list(history)

        try:
            # Send typing indicator
            await update.message.chat.send_action("typing")

            # Call LLM to interpret intent
            loop = asyncio.get_event_loop()
            llm_response = await loop.run_in_executor(
                None,
                partial(_call_openrouter_sync, messages, _NL_SYSTEM_PROMPT),
            )
            logger.info("NL router response: %s", llm_response[:200])
        except Exception as e:
            logger.error("NL router LLM call failed: %s", traceback.format_exc())
            await update.message.reply_text(
                f"Sorry, I couldn't process that right now: {str(e)[:200]}"
            )
            return

        # Check if the LLM returned an action
        action_data = _parse_action(llm_response)

        if action_data:
            action = action_data["action"]
            params = action_data.get("params", {})
            logger.info("NL action detected: %s params=%s", action, params)

            # Handle special non-graph actions
            if action == "cost_report":
                await cmd_cost_report(update, context)
                history.append({"role": "assistant", "content": "[Ran cost report]"})
                return
            if action == "status":
                await cmd_status(update, context)
                history.append({"role": "assistant", "content": "[Ran status check]"})
                return

            # Run the agent task
            await update.message.reply_text(f"On it \u2014 running {action.replace('_', ' ')}...")

            result = {}
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, partial(_run_graph_sync, action, **params)
                )
            except Exception as e:
                logger.error("Agent task %s failed: %s", action, traceback.format_exc())
                await update.message.reply_text(f"That didn't work: {str(e)[:300]}")
                history.append({"role": "assistant", "content": f"[Task {action} failed: {str(e)[:100]}]"})
                return

            # Format and send results
            formatted = await _format_task_result(action, result)

            # Telegram 4096 char limit
            for i in range(0, len(formatted), 4000):
                await update.message.reply_text(formatted[i:i + 4000])

            # Store a summary in conversation history
            summary = formatted[:300] + ("..." if len(formatted) > 300 else "")
            history.append({"role": "assistant", "content": summary})

        else:
            # Pure conversational response
            await update.message.reply_text(llm_response)
            history.append({"role": "assistant", "content": llm_response})

    except Exception as e:
        logger.error("handle_natural_language crashed: %s", traceback.format_exc())
        try:
            await update.message.reply_text(f"Something went wrong: {str(e)[:300]}")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Bot setup and main
# ---------------------------------------------------------------------------


async def post_init(application: Application) -> None:
    """Set bot commands in the Telegram menu after startup."""
    commands = [
        BotCommand("start", "Show help and available commands"),
        BotCommand("status", "System health check"),
        BotCommand("keyword_research", "Find keyword opportunities"),
        BotCommand("content_gap", "Find content gaps vs competitors"),
        BotCommand("content_brief", "Generate a content brief"),
        BotCommand("discover_prospects", "Find backlink prospects"),
        BotCommand("score_prospects", "Score enriched prospects"),
        BotCommand("generate_emails", "Generate outreach emails"),
        BotCommand("cost_report", "LLM spend this week"),
        BotCommand("rank_report", "Ranking report"),
        BotCommand("weekly_report", "Full weekly SEO report"),
        BotCommand("outreach_report", "Outreach summary"),
    ]
    await application.bot.set_my_commands(commands)
    logger.info("Bot commands registered with Telegram")


def main() -> None:
    """Start the Telegram bot with long polling."""
    # Strip all whitespace/newlines — Railway env vars often have trailing \n
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    token = "".join(token.split())  # removes ALL whitespace including embedded \n
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN is not set")
        sys.exit(1)

    logger.info("Starting RalfSEObot with token length=%d...", len(token))

    app = Application.builder().token(token).post_init(post_init).build()

    # Register command handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("keyword_research", cmd_keyword_research))
    app.add_handler(CommandHandler("content_gap", cmd_content_gap))
    app.add_handler(CommandHandler("content_brief", cmd_content_brief))
    app.add_handler(CommandHandler("discover_prospects", cmd_discover_prospects))
    app.add_handler(CommandHandler("score_prospects", cmd_score_prospects))
    app.add_handler(CommandHandler("generate_emails", cmd_generate_emails))
    app.add_handler(CommandHandler("cost_report", cmd_cost_report))
    app.add_handler(CommandHandler("rank_report", cmd_rank_report))
    app.add_handler(CommandHandler("weekly_report", cmd_weekly_report))
    app.add_handler(CommandHandler("outreach_report", cmd_outreach_report))

    # Catch unknown commands
    app.add_handler(MessageHandler(filters.COMMAND, cmd_unknown))

    # Natural language handler — catches all non-command text messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_natural_language))

    # Global error handler — catches all unhandled exceptions
    async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        logger.error("Unhandled exception: %s", context.error, exc_info=context.error)
        if isinstance(update, Update) and update.message:
            try:
                await update.message.reply_text(
                    f"Internal error: {str(context.error)[:300]}"
                )
            except Exception:
                pass

    app.add_error_handler(error_handler)

    logger.info("RalfSEObot is running — polling for messages...")
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    # Startup diagnostics for Railway deploy logs
    raw_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    clean_token = raw_token.strip()
    print("[BOOT] telegram_bot.py starting...")
    print(f"[BOOT] Python: {sys.version}")
    print(f"[BOOT] TELEGRAM_BOT_TOKEN: len={len(raw_token)} stripped={len(clean_token)} has_newline={chr(10) in raw_token}")
    print(f"[BOOT] OPENROUTER_API_KEY: {'SET' if os.environ.get('OPENROUTER_API_KEY') else 'MISSING'}")
    print(f"[BOOT] SUPABASE_URL: {'SET' if os.environ.get('SUPABASE_URL') else 'MISSING'}")
    print(f"[BOOT] SUPABASE_SERVICE_KEY: {'SET' if os.environ.get('SUPABASE_SERVICE_KEY') else 'MISSING'}")
    print(f"[BOOT] CWD: {os.getcwd()}")
    sys.stdout.flush()
    main()
