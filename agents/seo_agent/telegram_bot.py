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

_NL_SYSTEM_PROMPT = """You are Ralf, a proactive SEO agent for three interconnected websites:

1. kitchensdirectory.co.uk — Directory of 159+ verified UK handmade kitchen makers. 11 styles, 4 budget tiers. Monetised via ads + leads.
2. freeroomplanner.com — Free browser floor planner. 30+ furniture items, snap-to-grid, PNG export. No sign-up. Monetised via lead capture.
3. kitchencostestimator.com — Kitchen cost calculator for UK/US/Canada. 68 cost items, 26 multipliers. Monetised via high-intent leads.

Existing blog posts on freeroomplanner.com:
- bathroom-planning-mistakes, brief-your-builder, extension-planning,
  kitchen-layout-guide, open-plan-living, small-bedroom-layouts

All three sites cross-link to each other.

SITE STATUS:
- freeroomplanner.com = ACTIVE (primary focus, all SEO work goes here first)
- kitchensdirectory.co.uk = UPCOMING (not doing active SEO yet, don't run tasks for it unless asked)
- kitchencostestimator.com = UPCOMING (not doing active SEO yet, don't run tasks for it unless asked)

When running tasks for "all" sites, only include ACTIVE sites unless the user explicitly asks for all.

PERSONALITY: You're Ralf — direct, warm, and competent. Talk like a sharp colleague, not a robot.
- Be conversational and natural. No bullet-point brain dumps. No corporate speak.
- Write like you're texting a friend who happens to be your boss. Casual but professional.
- When the user says to do something, just do it. Don't narrate your process or list options.
- After finishing a task, briefly mention what you did and what you're doing next. One or two sentences, not a report.
- You're the SEO lead. The user is the founder. You bring updates and momentum, not questions.
- If there's an obvious next step, just do it. If you genuinely need a decision, ask concisely.
- NEVER end with "What's the priority?" or "Say the word." If you know what to do, do it.
- Don't say things like "I'm treating this as a brief acknowledgement" — just be normal. Say "Cheers!" or "On it." or "Nice one."
- NEVER end a message with "What's the priority?" or "Want me to do X or Y?" If you know what the next step should be, just do it and tell the user what you did.
- When reporting status, ALWAYS follow with what you're doing next. Not "want me to?" — just "I'm doing X next."
- If the user asks "what have you been doing?", give a concise summary and IMMEDIATELY state your next action. Don't wait for permission.
- When the user points out a problem ("why do you keep doing X?"), acknowledge it directly. Don't deflect or claim it didn't happen. Say "you're right, I did X — here's why and here's what I'm changing."

ACTION FORMAT: Return ONLY raw JSON (no markdown, no code fences, no explanation):
{"action": "<type>", "params": {"key": "value"}}

NEVER wrap actions in ```json``` code blocks. NEVER add text before or after the JSON.
If you need to run an action, your ENTIRE response must be the JSON object and nothing else.
If you want to chat, respond in plain text with NO JSON.

Actions:
- keyword_research: {target_site, seed_keyword?}
- content_gap: {target_site}
- content_brief: {target_site, selected_keyword}
- discover_prospects: {target_site}
- score_prospects: {target_site:"all"}
- generate_emails: {target_site:"all"}
- rank_report: {target_site}
- weekly_report: {target_site:"all"}
- outreach_report: {target_site:"all"}
- cost_report: (no params)
- status: (no params)
- web_search: {query} — search the internet
- recall: {topic} — query our database for past results
- list_blogs: {site} — list existing blog posts from GitHub
- publish_blog: {site, title, keyword} — write and publish a blog post
- store_content: {site, content_list} — save existing site content to database
- dashboard: (no params) — full overview of all data, pipeline, and progress
- prospect_pipeline: (no params) — show prospect CRM pipeline by stage
- followups: (no params) — show prospects needing follow-up
- ranking_movers: {target_site} — show biggest ranking changes (winners/losers)
- track_rankings: {target_site} — snapshot current rankings from Ahrefs
- learn: {topic} — research a topic online and add to the knowledge base
- knowledge: {query} — search the knowledge base for specific SEO/AEO info
- journal: {category?} — write a reflective blog post on Ralf's personal blog
- audit: {target_site} — run full SEO audit (scores 0-100 across 8 categories)
- audit_page: {url} — audit a specific URL
- crm_pipeline: (no params) — show the CRM outreach pipeline (companies and designers)
- crm_search: {query} — search the CRM for contacts by name, company, city, or category
- crm_followups: (no params) — show CRM contacts needing follow-up
- import_makers: {city?} — import kitchen makers from the directory into the CRM
- scrape_companies: {country, category?} — use Firecrawl to find and extract kitchen/bathroom companies (country: UK, US, or CA)
- scrape_all: (no params) — run full scrape across all countries
- scrape_status: (no params) — show CRM contact counts by country and category

Site keys: kitchensdirectory, freeroomplanner, kitchen_estimator, ralf_seo, all

CRITICAL RULES:
0. If the user sends casual feedback ("great job", "nice", "thanks", "ok", "cool", "perfect",
   "looking forward to it", emoji, or brief acknowledgement), respond naturally like a human would.
   "Cheers!", "Thanks!", "Nice one.", "Glad you like it." Do NOT start a new task.
1. NEVER output JSON inside markdown code blocks. Raw JSON only.
2. Be proactive. After completing a task, EXECUTE the logical next step immediately.
   Example: after keyword research, create content briefs for the top 5 keywords, then start writing.
   DON'T ask permission for obvious next steps. DON'T end with "What should I focus on?" or "What's the priority?"
   Just do it and report what you did.
3. When the user says "store" or "save" or "remember" content — that means write it to the database using store_content. Do NOT search the web.
   When the user gives workflow instructions ("save results", "don't use the API so much", "cache results", "be more efficient", etc.), acknowledge the instruction and confirm what you'll do differently. Do NOT interpret workflow guidance as a task request.
4. To review OUR OWN sites, use list_blogs (for blog posts) or recall (for database). Do NOT use web_search to look at our own sites — Tavily returns competitor content, not ours.
5. Keep responses SHORT. This is Telegram. Max 3-4 short paragraphs.
6. ALWAYS follow through. Never say "give me a sec" or "let me pull that" without actually returning an action. If you need to do something, return the action JSON.
7. You have full conversation history and a Supabase database. Never say you can't remember or don't have memory.
8. When mapping site names: "room planner" / "freeroomplanner" = freeroomplanner, "directory" / "kitchens" = kitchensdirectory, "estimator" / "cost" = kitchen_estimator.
9. Default to freeroomplanner if context is about room/floor planning, kitchensdirectory if about kitchen makers/companies.
10. When the user gives you INSTRUCTIONS about how to work (e.g., "save results before calling APIs", "don't burn API tokens", "be more careful with X"), acknowledge the instruction conversationally. Do NOT re-run a task. Just confirm you understand and will change your approach.
    Example: User says "save the keyword results so you don't keep using the Ahrefs API" → respond "Got it — I'll check our cached results before hitting Ahrefs from now on. The keywords I just found are already saved." Do NOT run keyword_research again.
11. CONTENT DIVERSITY: Never write 2+ blog posts on the same topic in a row. Mix it up — if we just wrote about kitchens, the next post should be about bathrooms, room planning, bedrooms, or extensions. Check what was recently published before choosing the next topic.
12. PRIVACY: Never include personal information (owner's name, email addresses, API keys, tokens, chat IDs, project IDs, or internal URLs) in any content that will be published publicly — blog posts, outreach emails, or any external-facing text. Refer to the owner as "the founder" in public content. You may mention website domains and service names.

OUTREACH STRATEGY (use this when discussing backlinks, outreach, or prospecting):
- Kitchen/bathroom providers: PARTNERSHIP approach. Offer free room planner embed for their website. Their customers plan before visiting = better conversion for them.
- Home interior bloggers: CONTENT COLLABORATION. Offer exclusive cost data, guest post exchange, tool features. Suggest specific content ideas.
- Home improvement influencers: INFLUENCER COLLAB. Offer free tools for their audience, room planning challenges, cross-promotion.
- Resource page owners: RESOURCE INCLUSION. Brief, respectful pitch. Free tool, no catch, useful for their readers.
- PR journalists: STORY ANGLE. Lead with data ('what kitchens really cost in 2026'), offer expert comment.

All outreach must lead with THEIR benefit, not ours. We're collaborators, not beggars. Keep emails under 150 words. Sign off as Ben, not Ralf.
Monthly goals: 20 provider partnerships, 15 blogger collabs, 5 influencers, 10 resource pages, 5 PR pitches.
"""


def _build_strategy_context() -> str:
    """Build a dynamic strategy context from the CRM dashboard."""
    try:
        from agents.seo_agent.tools.crm_tools import get_dashboard_summary
        from agents.seo_agent.strategy import generate_next_steps, get_strategy_summary

        dash = get_dashboard_summary()

        next_steps = generate_next_steps(
            existing_keywords=dash["keywords_discovered"],
            existing_content=dash["content_pieces"],
            existing_prospects=dash["prospects_total"],
            existing_gaps=dash["content_gaps"],
        )

        context = f"\n\nCURRENT STATE: {dash['keywords_discovered']} keywords, {dash['content_pieces']} content, {dash['content_gaps']} gaps, {dash['prospects_total']} prospects, {dash['rankings_tracked']} rankings tracked."
        if dash.get("prospect_pipeline"):
            context += f"\nPipeline: {dash['prospect_pipeline']}"
        context += f"\nWeekly spend: ${dash['weekly_spend']:.4f}"
        context += "\n\nPRIORITISED NEXT STEPS (suggest these proactively):"
        for i, step in enumerate(next_steps, 1):
            context += f"\n{i}. {step}"
        context += f"\n\n{get_strategy_summary()}"

        # Inject knowledge base summary
        try:
            from agents.seo_agent.tools.knowledge_tools import get_knowledge_summary
            kb_summary = get_knowledge_summary()
            if kb_summary:
                context += kb_summary
        except Exception:
            pass

        return context
    except Exception:
        return ""


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
        text = response.choices[0].message.content or ""
    else:
        client = _get_anthropic_client()
        response = client.messages.create(
            model=_resolve_model_id(_HAIKU),
            max_tokens=500,
            system=system,
            messages=messages,
        )
        text = "".join(b.text for b in response.content if b.type == "text")

    # Log intent classification cost
    try:
        from agents.seo_agent.tools.supabase_tools import log_llm_cost
        log_llm_cost(
            task_type="intent_classification",
            model=_resolve_model_id(_HAIKU),
            input_tokens=0,
            output_tokens=0,
            cached_tokens=0,
            cost_usd=0.001,
            site="",
        )
    except Exception:
        pass

    return text


def _parse_action(text: str) -> dict | None:
    """Try to extract a JSON action block from the LLM response.

    Handles: raw JSON, JSON in code blocks, JSON mixed with text.
    """
    text = text.strip()

    # 1. Try direct JSON parse (ideal case)
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "action" in data:
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Try extracting from markdown code blocks
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

    # 3. Try finding JSON object anywhere in the text
    # Find the first { and try parsing from there
    idx = text.find('{"action"')
    if idx == -1:
        idx = text.find("{'action")
    if idx >= 0:
        # Find matching closing brace
        depth = 0
        for i in range(idx, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(text[idx:i + 1])
                        if isinstance(data, dict) and "action" in data:
                            return data
                    except (json.JSONDecodeError, ValueError):
                        pass
                    break

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


def _recall_from_database(topic: str) -> str:
    """Query Supabase for past results matching a topic."""
    from agents.seo_agent.tools.supabase_tools import query_table

    results_parts = []

    # Check keywords
    if any(w in topic.lower() for w in ["keyword", "opportunity", "research", "all", "everything"]):
        keywords = query_table("seo_keyword_opportunities", limit=20)
        if keywords:
            results_parts.append(f"Keyword opportunities ({len(keywords)} in DB):")
            for kw in keywords[:15]:
                results_parts.append(f"  - {kw.get('keyword')} | vol:{kw.get('volume')} KD:{kw.get('kd')} | site:{kw.get('target_site')}")

    # Check content gaps
    if any(w in topic.lower() for w in ["gap", "content", "all", "everything"]):
        gaps = query_table("seo_content_gaps", limit=20)
        if gaps:
            results_parts.append(f"\nContent gaps ({len(gaps)} in DB):")
            for g in gaps[:15]:
                results_parts.append(f"  - {g.get('keyword')} | vol:{g.get('volume')} | stage:{g.get('funnel_stage')} | site:{g.get('target_site')}")

    # Check prospects
    if any(w in topic.lower() for w in ["prospect", "backlink", "outreach", "link", "all", "everything"]):
        prospects = query_table("seo_backlink_prospects", limit=20)
        if prospects:
            results_parts.append(f"\nBacklink prospects ({len(prospects)} in DB):")
            for p in prospects[:15]:
                results_parts.append(f"  - {p.get('domain')} | DR:{p.get('dr')} | status:{p.get('status')} | site:{p.get('target_site')}")

    # Check briefs
    if any(w in topic.lower() for w in ["brief", "content brief", "all", "everything"]):
        briefs = query_table("seo_content_briefs", limit=10)
        if briefs:
            results_parts.append(f"\nContent briefs ({len(briefs)} in DB):")
            for b in briefs[:10]:
                results_parts.append(f"  - {b.get('title', b.get('keyword'))} | site:{b.get('target_site')}")

    # Check cost log
    if any(w in topic.lower() for w in ["cost", "spend", "budget", "llm"]):
        costs = query_table("llm_cost_log", limit=20, order_by="created_at", order_desc=True)
        if costs:
            total = sum(c.get("cost_usd", 0) for c in costs)
            results_parts.append(f"\nLLM costs (last {len(costs)} calls, total: ${total:.4f}):")
            by_task = {}
            for c in costs:
                t = c.get("task_type", "unknown")
                by_task[t] = by_task.get(t, 0) + c.get("cost_usd", 0)
            for t, cost in sorted(by_task.items(), key=lambda x: -x[1]):
                results_parts.append(f"  - {t}: ${cost:.4f}")

    if not results_parts:
        return f"No data found in the database for topic: {topic}. Try running a task first (keyword research, content gap, etc.)."

    return "\n".join(results_parts)


def _generate_blog_post(site: str, title: str, keyword: str) -> dict[str, str]:
    """Generate a full SEO-optimised blog post using Claude (Sonnet)."""
    from agents.seo_agent.tools.llm_router import (
        _get_openrouter_client, _use_openrouter, _resolve_model_id, _SONNET,
        _get_anthropic_client,
    )
    from agents.seo_agent.config import SITE_PROFILES

    profile = SITE_PROFILES.get(site, {})
    site_desc = profile.get("description", "")
    audience = profile.get("target_audience", "")

    prompt = f"""Write a complete, SEO-optimised blog post for the website {profile.get('domain', site)}.

Site context: {site_desc}
Target audience: {audience}

Topic/Title: {title}
Target keyword: {keyword}

Requirements:
- Write 1500-2500 words of high-quality, original content
- Use the target keyword naturally 3-5 times
- Include an engaging introduction
- Use H2 and H3 subheadings (as HTML tags: <h2>, <h3>)
- Include bullet points and numbered lists where appropriate
- Add a FAQ section with 3-5 questions using <h3> tags
- Include internal linking opportunities (mention related tools/pages on the site)
- Write in a helpful, authoritative but approachable tone
- Use UK English spelling
- Format as HTML body content (no <html>, <head>, or <body> tags — just the article content)
- Include <p>, <h2>, <h3>, <ul>, <li>, <ol>, <strong>, <em> tags as needed
- Do NOT include any markdown formatting — use only HTML
- NEVER include personal names, email addresses, API keys, tokens, or internal identifiers
- Refer to the site owner as "the team" or use passive voice — never use personal names

Return your response in this exact JSON format:
{{{{
  "title": "The SEO-optimised title",
  "meta_description": "A compelling 150-160 character meta description",
  "content": "<h2>First section</h2><p>Content here...</p>..."
}}}}

Return ONLY the JSON, no other text."""

    messages = [{"role": "user", "content": prompt}]

    if _use_openrouter():
        client = _get_openrouter_client()
        response = client.chat.completions.create(
            model=_resolve_model_id(_SONNET),
            max_tokens=4000,
            messages=[{"role": "system", "content": "You are an expert SEO content writer."}, *messages],
        )
        text = response.choices[0].message.content or ""
    else:
        client = _get_anthropic_client()
        response = client.messages.create(
            model=_resolve_model_id(_SONNET),
            max_tokens=4000,
            system="You are an expert SEO content writer.",
            messages=messages,
        )
        text = "".join(b.text for b in response.content if b.type == "text")

    # Log blog generation cost
    try:
        from agents.seo_agent.tools.supabase_tools import log_llm_cost
        log_llm_cost(
            task_type="blog_generation",
            model=_resolve_model_id(_SONNET),
            input_tokens=0,
            output_tokens=0,
            cached_tokens=0,
            cost_usd=0.02,
            site=site,
        )
    except Exception:
        pass

    # Parse the JSON response — robust extraction with multiple fallbacks
    data = None
    # 1. Try direct parse
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Try extracting from markdown code block
    if data is None and "```" in text:
        for block in text.split("```"):
            block = block.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            try:
                data = json.loads(block)
                break
            except json.JSONDecodeError:
                continue

    # 3. Try regex extraction of content field from partial/truncated JSON
    if data is None:
        import re as _re
        title_m = _re.search(r'"title"\s*:\s*"([^"]+)"', text)
        meta_m = _re.search(r'"meta_description"\s*:\s*"([^"]+)"', text)
        content_m = _re.search(r'"content"\s*:\s*"(.+?)"\s*[,}]', text, _re.DOTALL)
        if content_m:
            extracted = content_m.group(1).replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
            data = {
                "title": title_m.group(1) if title_m else title,
                "meta_description": meta_m.group(1) if meta_m else "",
                "content": extracted,
            }
            logger.info("Extracted blog content via regex fallback (%d chars)", len(extracted))

    # 4. Final fallback — use raw text but strip any JSON/markdown artifacts
    if data is None:
        clean = text
        clean = _re.sub(r'^```json\s*', '', clean)
        clean = _re.sub(r'```\s*$', '', clean)
        clean = _re.sub(r'^\{[^}]*"content"\s*:\s*"', '', clean)
        clean = clean.rstrip('"}` \n')
        data = {"title": title, "meta_description": "", "content": clean}
        logger.warning("Blog post JSON parse failed — used raw text fallback")

    return {
        "title": data.get("title", title),
        "meta_description": data.get("meta_description", ""),
        "content": data.get("content", text),
    }


def _run_web_search(query: str) -> list[dict]:
    """Run a Tavily web search synchronously."""
    from agents.seo_agent.tools.web_search_tools import search
    return search(query, max_results=8)


def _summarise_search_results(
    query: str, results: list[dict], conversation: list[dict]
) -> str:
    """Use Claude to summarise search results into a natural response."""
    results_text = "\n\n".join(
        f"Title: {r.get('title', 'N/A')}\nURL: {r.get('url', '')}\nContent: {r.get('content', '')[:300]}"
        for r in results[:6]
    )
    messages = conversation + [{
        "role": "user",
        "content": f"Based on these web search results for '{query}', give me a concise summary. "
        f"Include key findings and relevant URLs.\n\nSearch Results:\n{results_text}"
    }]
    return _call_openrouter_sync(messages, _NL_SYSTEM_PROMPT)


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

            # Build strategy-aware system prompt
            strategy_context = await asyncio.get_event_loop().run_in_executor(
                None, _build_strategy_context
            )
            full_prompt = _NL_SYSTEM_PROMPT + strategy_context

            # Call LLM to interpret intent
            loop = asyncio.get_event_loop()
            llm_response = await loop.run_in_executor(
                None,
                partial(_call_openrouter_sync, messages, full_prompt),
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

            # Strip JSON from visible response — send only conversational text
            json_start = llm_response.find('{"action"')
            if json_start == -1:
                json_start = llm_response.find("{'action")
            pre_action_text = llm_response[:json_start].strip() if json_start > 0 else ""

            if pre_action_text and len(pre_action_text) > 10:
                await update.message.reply_text(pre_action_text)
                history.append({"role": "assistant", "content": pre_action_text[:200]})

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
            if action == "recall":
                topic = params.get("topic", user_text)
                try:
                    db_results = await asyncio.get_event_loop().run_in_executor(
                        None, partial(_recall_from_database, topic)
                    )
                    # Send raw data to Claude for a natural summary
                    summary = await asyncio.get_event_loop().run_in_executor(
                        None, partial(_call_openrouter_sync,
                            list(history) + [{"role": "user", "content": f"Here's what I found in our database:\n\n{db_results}\n\nSummarise this for me naturally."}],
                            _NL_SYSTEM_PROMPT)
                    )
                    await update.message.reply_text(summary)
                    history.append({"role": "assistant", "content": summary[:300]})
                except Exception as e:
                    logger.error("Recall failed: %s", traceback.format_exc())
                    await update.message.reply_text(f"Couldn't retrieve data: {str(e)[:200]}")
                return
            if action == "dashboard":
                try:
                    from agents.seo_agent.tools.crm_tools import get_dashboard_summary
                    dash = await asyncio.get_event_loop().run_in_executor(None, get_dashboard_summary)
                    lines = ["SEO Dashboard:\n"]
                    lines.append(f"Keywords: {dash['keywords_discovered']} discovered, {dash['keywords_cached']} cached")
                    lines.append(f"Content: {dash['content_pieces']} pieces, {dash['content_gaps']} gaps")
                    lines.append(f"Prospects: {dash['prospects_total']} total")
                    if dash.get('prospect_pipeline'):
                        lines.append("Pipeline: " + ", ".join(f"{k}:{v}" for k, v in dash['prospect_pipeline'].items()))
                    lines.append(f"Rankings tracked: {dash['rankings_tracked']} keywords")
                    lines.append(f"Weekly spend: ${dash['weekly_spend']:.4f}")
                    lines.append("\nPer site:")
                    for site, data in dash.get('sites', {}).items():
                        lines.append(f"  {site}: {data['keywords']}kw / {data['content']}content / {data['prospects']}prospects")
                    # Add CRM stats
                    try:
                        from agents.seo_agent.tools.crm_tools import get_crm_pipeline
                        crm = get_crm_pipeline()
                        crm_total = sum(len(v) for v in crm.values())
                        if crm_total > 0:
                            lines.append(f"\nCRM: {crm_total} contacts")
                            for crm_status, contacts in crm.items():
                                if contacts:
                                    lines.append(f"  {crm_status}: {len(contacts)}")
                    except Exception:
                        pass
                    msg = "\n".join(lines)
                    await update.message.reply_text(msg)
                    history.append({"role": "assistant", "content": msg[:300]})
                except Exception as e:
                    await update.message.reply_text(f"Dashboard error: {str(e)[:200]}")
                return
            if action == "prospect_pipeline":
                try:
                    from agents.seo_agent.tools.crm_tools import get_prospect_pipeline
                    pipeline = await asyncio.get_event_loop().run_in_executor(None, get_prospect_pipeline)
                    lines = ["Prospect Pipeline:\n"]
                    total = 0
                    for stage, prospects_list in pipeline.items():
                        if prospects_list:
                            total += len(prospects_list)
                            lines.append(f"\n{stage}: {len(prospects_list)}")
                            for p in prospects_list[:10]:
                                dr = p.get('dr', 0)
                                domain = p.get('domain', '?')
                                method = p.get('discovery_method', '?')
                                traffic = p.get('monthly_traffic', 0)
                                title = p.get('page_title', '')[:40]
                                detail = f"  - {domain} (DR:{dr}"
                                if traffic:
                                    detail += f", {traffic:,} visits/mo"
                                detail += f", via {method})"
                                if title:
                                    detail += f"\n    {title}"
                                lines.append(detail)
                            if len(prospects_list) > 10:
                                lines.append(f"  ...and {len(prospects_list) - 10} more")
                    if total == 0:
                        lines.append("Empty — run discover_prospects first.")
                    else:
                        lines.append(f"\nTotal: {total} prospects")
                    await update.message.reply_text("\n".join(lines))
                except Exception as e:
                    await update.message.reply_text(f"Pipeline error: {str(e)[:200]}")
                return
            if action == "followups":
                try:
                    from agents.seo_agent.tools.crm_tools import get_prospects_needing_followup
                    needs = await asyncio.get_event_loop().run_in_executor(None, get_prospects_needing_followup)
                    if needs:
                        lines = [f"{len(needs)} prospects need follow-up:\n"]
                        for p in needs[:10]:
                            lines.append(f"- {p.get('domain')} | last contact: {p.get('last_contacted_at', 'N/A')[:10]}")
                        await update.message.reply_text("\n".join(lines))
                    else:
                        await update.message.reply_text("No prospects need follow-up right now.")
                except Exception as e:
                    await update.message.reply_text(f"Follow-up check error: {str(e)[:200]}")
                return
            if action == "crm_pipeline":
                try:
                    from agents.seo_agent.tools.crm_tools import get_crm_pipeline
                    pipeline = await asyncio.get_event_loop().run_in_executor(None, get_crm_pipeline)
                    lines = ["CRM Pipeline:\n"]
                    total = 0
                    for status, contacts in pipeline.items():
                        if contacts:
                            total += len(contacts)
                            lines.append(f"\n{status}: {len(contacts)}")
                            for c in contacts[:8]:
                                name = c.get("company_name", "?")
                                cat = c.get("category", "")
                                city = c.get("city", "")
                                email = c.get("email", "")
                                detail = f"  - {name}"
                                if cat:
                                    detail += f" ({cat})"
                                if city:
                                    detail += f" — {city}"
                                if email:
                                    detail += f" [{email}]"
                                lines.append(detail)
                            if len(contacts) > 8:
                                lines.append(f"  ...and {len(contacts) - 8} more")
                    if total == 0:
                        lines.append("No contacts yet. Use import_makers or add contacts via discover_prospects.")
                    else:
                        lines.append(f"\nTotal: {total} contacts")
                    await update.message.reply_text("\n".join(lines))
                except Exception as e:
                    await update.message.reply_text(f"CRM error: {str(e)[:200]}")
                return
            if action == "crm_search":
                query_text = params.get("query", user_text)
                try:
                    from agents.seo_agent.tools.crm_tools import search_crm_contacts
                    results = await asyncio.get_event_loop().run_in_executor(
                        None, partial(search_crm_contacts, query_text)
                    )
                    if results:
                        lines = [f"Found {len(results)} contacts:\n"]
                        for c in results[:10]:
                            name = c.get("company_name", "?")
                            cat = c.get("category", "")
                            city = c.get("city", "")
                            crm_status = c.get("outreach_status", "")
                            lines.append(f"- {name} ({cat}) — {city} [{crm_status}]")
                        await update.message.reply_text("\n".join(lines))
                    else:
                        await update.message.reply_text(f"No contacts found for '{query_text}'.")
                except Exception as e:
                    await update.message.reply_text(f"Search failed: {str(e)[:200]}")
                return
            if action == "crm_followups":
                try:
                    from agents.seo_agent.tools.crm_tools import get_crm_contacts_needing_followup
                    needs = await asyncio.get_event_loop().run_in_executor(None, get_crm_contacts_needing_followup)
                    if needs:
                        lines = [f"{len(needs)} contacts need follow-up:\n"]
                        for c in needs[:10]:
                            name = c.get("company_name", "?")
                            last = c.get("last_contacted_at", "never")
                            if isinstance(last, str) and len(last) > 10:
                                last = last[:10]
                            lines.append(f"- {name} (last contact: {last})")
                        await update.message.reply_text("\n".join(lines))
                    else:
                        await update.message.reply_text("No contacts need follow-up right now.")
                except Exception as e:
                    await update.message.reply_text(f"Follow-up check failed: {str(e)[:200]}")
                return
            if action == "import_makers":
                city = params.get("city")
                await update.message.reply_text(f"Importing kitchen makers{' from ' + city if city else ''}...")
                try:
                    from agents.seo_agent.tools.crm_tools import import_kitchen_makers_to_crm
                    count = await asyncio.get_event_loop().run_in_executor(
                        None, partial(import_kitchen_makers_to_crm, city)
                    )
                    await update.message.reply_text(f"Imported {count} kitchen makers into the CRM.")
                    history.append({"role": "assistant", "content": f"Imported {count} makers into CRM"})
                except Exception as e:
                    await update.message.reply_text(f"Import failed: {str(e)[:200]}")
                return
            if action == "ranking_movers":
                site = params.get("target_site", "freeroomplanner")
                try:
                    from agents.seo_agent.tools.crm_tools import get_ranking_movers
                    movers = await asyncio.get_event_loop().run_in_executor(None, partial(get_ranking_movers, site))
                    lines = [f"Ranking movers for {site}:\n"]
                    if movers.get('winners'):
                        lines.append("Winners:")
                        for w in movers['winners'][:5]:
                            lines.append(f"  +{w.get('change',0)} {w.get('keyword')} (pos {w.get('position')})")
                    if movers.get('losers'):
                        lines.append("Losers:")
                        for l in movers['losers'][:5]:
                            lines.append(f"  {l.get('change',0)} {l.get('keyword')} (pos {l.get('position')})")
                    if len(lines) == 1:
                        lines.append("No ranking data yet — run track_rankings first.")
                    await update.message.reply_text("\n".join(lines))
                except Exception as e:
                    await update.message.reply_text(f"Ranking movers error: {str(e)[:200]}")
                return
            if action == "track_rankings":
                site = params.get("target_site", "freeroomplanner")
                await update.message.reply_text(f"Snapshotting rankings for {site} from Ahrefs...")
                try:
                    from agents.seo_agent.tools.ahrefs_tools import get_organic_keywords
                    from agents.seo_agent.tools.crm_tools import snapshot_our_rankings
                    from agents.seo_agent.config import SITE_PROFILES
                    profile = SITE_PROFILES.get(site, {})
                    domain = profile.get("domain", "")
                    if domain:
                        rankings = await asyncio.get_event_loop().run_in_executor(
                            None, partial(get_organic_keywords.invoke, domain)
                        )
                        saved = await asyncio.get_event_loop().run_in_executor(
                            None, partial(snapshot_our_rankings, site, rankings)
                        )
                        await update.message.reply_text(f"Saved {saved} ranking positions for {site}. Use 'ranking movers' to see changes over time.")
                    else:
                        await update.message.reply_text(f"No domain configured for {site}.")
                except Exception as e:
                    logger.error("Track rankings failed: %s", traceback.format_exc())
                    await update.message.reply_text(f"Ranking snapshot failed: {str(e)[:200]}")
                return
            if action in ("audit", "audit_page"):
                site = params.get("target_site", "freeroomplanner")
                url = params.get("url", "")
                await update.message.reply_text(f"Running SEO audit{' for ' + url if url else ' for ' + site}... this takes a minute.")
                try:
                    from agents.seo_agent.tools.seo_audit import audit_site, audit_page, format_audit_report, save_audit
                    from agents.seo_agent.config import SITE_PROFILES

                    if url:
                        result = await asyncio.get_event_loop().run_in_executor(
                            None, partial(audit_page, url)
                        )
                    else:
                        profile = SITE_PROFILES.get(site, {})
                        domain = profile.get("domain", site)
                        result = await asyncio.get_event_loop().run_in_executor(
                            None, partial(audit_site, domain)
                        )
                        # Save for tracking
                        save_audit(site, result)

                    report = format_audit_report(result)
                    for i in range(0, len(report), 4000):
                        await update.message.reply_text(report[i:i + 4000])
                    history.append({"role": "assistant", "content": report[:300]})
                except Exception as e:
                    logger.error("Audit failed: %s", traceback.format_exc())
                    await update.message.reply_text(f"Audit failed: {str(e)[:300]}")
                return
            if action == "learn":
                topic = params.get("topic", user_text)
                await update.message.reply_text(f"Researching: {topic}...")
                try:
                    # Search the web for the topic
                    search_results = await asyncio.get_event_loop().run_in_executor(
                        None, partial(_run_web_search, f"SEO AEO {topic} 2026 best practices")
                    )
                    # Summarise and store
                    results_text = "\n".join(
                        f"- {r.get('title', '')}: {r.get('content', '')[:200]}" for r in search_results[:5]
                    )
                    summary = await asyncio.get_event_loop().run_in_executor(
                        None, partial(_call_openrouter_sync,
                            [{"role": "user", "content": f"Summarise these search results about '{topic}' into a concise knowledge entry (max 500 words). Focus on actionable SEO/AEO advice.\n\n{results_text}"}],
                            "You are an SEO expert. Summarise research into actionable knowledge.")
                    )
                    # Store in knowledge base
                    from agents.seo_agent.tools.knowledge_tools import store_knowledge
                    category = "aeo" if "aeo" in topic.lower() or "ai" in topic.lower() else "industry_trends"
                    import re
                    topic_slug = re.sub(r'[^a-z0-9_]', '_', topic.lower()[:80]).strip('_')
                    store_knowledge(category, topic_slug, summary)
                    await update.message.reply_text(f"Learned about '{topic}' and saved to knowledge base.\n\nKey points:\n{summary[:500]}")
                    history.append({"role": "assistant", "content": f"Learned about {topic}"})
                except Exception as e:
                    logger.error("Learn failed: %s", traceback.format_exc())
                    await update.message.reply_text(f"Research failed: {str(e)[:200]}")
                return
            if action == "knowledge":
                query = params.get("query", user_text)
                try:
                    from agents.seo_agent.tools.knowledge_tools import search_knowledge
                    results = await asyncio.get_event_loop().run_in_executor(
                        None, partial(search_knowledge, query)
                    )
                    if results:
                        lines = [f"Knowledge base ({len(results)} entries for '{query}'):"]
                        for r in results[:5]:
                            lines.append(f"\n[{r.get('category')}] {r.get('topic')}:")
                            lines.append(r.get('content', '')[:300])
                        msg = "\n".join(lines)
                        # Truncate for Telegram
                        for i in range(0, len(msg), 4000):
                            await update.message.reply_text(msg[i:i+4000])
                    else:
                        await update.message.reply_text(f"Nothing in the knowledge base about '{query}'. Want me to research it?")
                except Exception as e:
                    await update.message.reply_text(f"Knowledge search failed: {str(e)[:200]}")
                return
            if action == "store_content":
                site = params.get("site", "freeroomplanner")
                await update.message.reply_text(f"Cataloguing existing content for {site}...")
                try:
                    from agents.seo_agent.tools.github_tools import list_blog_posts
                    from agents.seo_agent.tools.supabase_tools import insert_record
                    posts = await asyncio.get_event_loop().run_in_executor(
                        None, partial(list_blog_posts, site)
                    )
                    stored = 0
                    for p in posts:
                        try:
                            insert_record("seo_content_briefs", {
                                "keyword": p["name"].replace(".html", "").replace(".mdx", "").replace("-", " "),
                                "target_site": site,
                                "title": p["name"].replace(".html", "").replace(".mdx", "").replace("-", " ").title(),
                                "file_path": p["path"],
                                "content_type": "existing_published",
                            })
                            stored += 1
                        except Exception:
                            pass
                    msg = f"Stored {stored} existing posts for {site} in the database. I'll avoid duplicating these topics in future content."
                    await update.message.reply_text(msg)
                    history.append({"role": "assistant", "content": msg})
                except Exception as e:
                    logger.error("Store content failed: %s", traceback.format_exc())
                    await update.message.reply_text(f"Failed to store content: {str(e)[:200]}")
                return
            if action == "publish_blog":
                site = params.get("site", "freeroomplanner")
                title = params.get("title", "")
                keyword = params.get("keyword", title)
                if not title and not keyword:
                    await update.message.reply_text("What should the blog post be about? Give me a title or keyword.")
                    return
                await update.message.reply_text(f"Writing blog post: {title or keyword}...\nThis may take a minute.")
                try:
                    # Step 1: Generate content via LLM
                    blog_content = await asyncio.get_event_loop().run_in_executor(
                        None, partial(_generate_blog_post, site, title or keyword, keyword)
                    )
                    # Step 2: Publish to GitHub
                    from agents.seo_agent.tools.github_tools import publish_blog_post
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, partial(publish_blog_post,
                            site=site,
                            title=blog_content["title"],
                            content=blog_content["content"],
                            meta_description=blog_content["meta_description"],
                        )
                    )
                    msg = (
                        f"Blog post published:\n\n"
                        f"Title: {blog_content['title']}\n"
                        f"URL: {result.get('published_url', 'N/A')}\n"
                        f"Commit: {result.get('commit_url', 'N/A')}\n\n"
                        f"It will be live after the site redeploys (1-2 minutes)."
                    )
                    await update.message.reply_text(msg)
                    history.append({"role": "assistant", "content": msg[:300]})
                except Exception as e:
                    logger.error("Blog publish failed: %s", traceback.format_exc())
                    await update.message.reply_text(f"Publishing failed: {str(e)[:300]}")
                return
            if action == "journal":
                category = params.get("category")
                await update.message.reply_text("Writing a journal entry...")
                try:
                    from agents.seo_agent.tools.reflection_engine import generate_reflective_post
                    from agents.seo_agent.tools.github_tools import publish_blog_post

                    post = await asyncio.get_event_loop().run_in_executor(
                        None, partial(generate_reflective_post, category=category)
                    )
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, partial(publish_blog_post,
                            site="ralf_seo",
                            title=post["title"],
                            content=post["content"],
                            meta_description=post["meta_description"],
                            category=post.get("category", "Field Report"),
                            what_i_learned=post.get("what_i_learned", []),
                        )
                    )
                    report = f"Published: {post['title']}\n\nCategory: {post.get('category')}\n\n"
                    report += f"URL: {result.get('published_url', 'N/A')}\n\n"
                    report += "What I learned:\n"
                    for item in post.get("what_i_learned", []):
                        report += f"\u2022 {item}\n"
                    await update.message.reply_text(report)
                    history.append({"role": "assistant", "content": report[:300]})
                except Exception as e:
                    logger.error("Journal failed: %s", traceback.format_exc())
                    await update.message.reply_text(f"Journal entry failed: {str(e)[:300]}")
                return
            if action == "list_blogs":
                site = params.get("site", "freeroomplanner")
                try:
                    from agents.seo_agent.tools.github_tools import list_blog_posts
                    posts = await asyncio.get_event_loop().run_in_executor(
                        None, partial(list_blog_posts, site)
                    )
                    if posts:
                        lines = [f"Blog posts for {site} ({len(posts)}):"]
                        for p in posts:
                            lines.append(f"\u2022 {p['name']}")
                        await update.message.reply_text("\n".join(lines))
                    else:
                        await update.message.reply_text(f"No blog posts found for {site}.")
                except Exception as e:
                    await update.message.reply_text(f"Failed to list posts: {str(e)[:200]}")
                return
            if action == "scrape_companies":
                country = params.get("country", "UK").upper()
                category = params.get("category", "kitchen_company")
                await update.message.reply_text(f"Starting Firecrawl scraper for {category} in {country}... this may take a few minutes.")
                try:
                    from agents.scraper_agent.tools.firecrawl_client import run_scraper
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, partial(run_scraper, country=country, category=category, max_queries=10)
                    )
                    lines = [
                        f"Scraping complete for {category} in {country}:\n",
                        f"Queries run: {result.get('queries_run', 0)}",
                        f"Company URLs found: {result.get('urls_found', 0)}",
                        f"Data extracted: {result.get('extracted', 0)}",
                        f"Added to CRM: {result.get('added_to_crm', 0)}",
                        f"Skipped (duplicates/invalid): {result.get('skipped', 0)}",
                    ]
                    if result.get("error"):
                        lines.append(f"\nError: {result['error']}")
                    await update.message.reply_text("\n".join(lines))
                    history.append({"role": "assistant", "content": f"Scraped {result.get('added_to_crm', 0)} {category} companies in {country}"})
                except Exception as e:
                    logger.error("Scraper failed: %s", traceback.format_exc())
                    await update.message.reply_text(f"Scraper failed: {str(e)[:300]}")
                return

            if action == "scrape_all":
                await update.message.reply_text("Starting full scrape across UK, US, and CA for kitchen + bathroom companies... this will take a while.")
                try:
                    from agents.scraper_agent.tools.firecrawl_client import run_full_scrape
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, partial(run_full_scrape, max_queries_per_country=5)
                    )
                    total = result.get("total_added", 0)
                    lines = [f"Full scrape complete — {total} companies added to CRM:\n"]
                    for key, data in result.items():
                        if key == "total_added":
                            continue
                        if isinstance(data, dict):
                            added = data.get("added_to_crm", 0)
                            found = data.get("urls_found", 0)
                            lines.append(f"  {key}: {added} added (from {found} found)")
                    await update.message.reply_text("\n".join(lines))
                    history.append({"role": "assistant", "content": f"Full scrape: {total} companies added"})
                except Exception as e:
                    logger.error("Full scrape failed: %s", traceback.format_exc())
                    await update.message.reply_text(f"Full scrape failed: {str(e)[:300]}")
                return

            if action == "scrape_status":
                try:
                    from agents.seo_agent.tools.crm_tools import get_crm_contacts
                    contacts = get_crm_contacts(limit=5000)
                    by_country: dict[str, int] = {}
                    by_category: dict[str, int] = {}
                    by_source: dict[str, int] = {}
                    for c in contacts:
                        ctry = c.get("country", "?")
                        by_country[ctry] = by_country.get(ctry, 0) + 1
                        cat = c.get("category", "?")
                        by_category[cat] = by_category.get(cat, 0) + 1
                        src = c.get("source", "?")
                        by_source[src] = by_source.get(src, 0) + 1

                    lines = [f"CRM: {len(contacts)} total contacts\n"]
                    if by_country:
                        lines.append("By country:")
                        for k, v in sorted(by_country.items(), key=lambda x: -x[1]):
                            lines.append(f"  {k}: {v}")
                    if by_category:
                        lines.append("\nBy category:")
                        for k, v in sorted(by_category.items(), key=lambda x: -x[1]):
                            lines.append(f"  {k}: {v}")
                    if by_source:
                        lines.append("\nBy source:")
                        for k, v in sorted(by_source.items(), key=lambda x: -x[1]):
                            lines.append(f"  {k}: {v}")
                    await update.message.reply_text("\n".join(lines))
                except Exception as e:
                    await update.message.reply_text(f"Status check failed: {str(e)[:200]}")
                return

            if action in ("web_search", "review_site"):
                query = params.get("query") or params.get("url") or user_text
                await update.message.reply_text(f"Searching: {query}...")
                try:
                    search_results = await asyncio.get_event_loop().run_in_executor(
                        None, partial(_run_web_search, query)
                    )
                    if not search_results:
                        msg = f"Search returned no results for '{query}'. Try a different query?"
                        await update.message.reply_text(msg)
                        history.append({"role": "assistant", "content": msg})
                        return
                    # Send results to Claude for a natural summary
                    summary = await asyncio.get_event_loop().run_in_executor(
                        None, partial(_summarise_search_results, query, search_results, list(history))
                    )
                    await update.message.reply_text(summary)
                    history.append({"role": "assistant", "content": f"[Searched: {query}] {summary[:250]}"})
                except Exception as e:
                    logger.error("Web search failed: %s", traceback.format_exc())
                    err_msg = f"Search failed: {str(e)[:200]}"
                    await update.message.reply_text(err_msg)
                    history.append({"role": "assistant", "content": f"[Search failed for: {query}] {err_msg}"})
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

    # Schedule autonomous heartbeat (every N hours)
    heartbeat_hours = int(os.getenv("HEARTBEAT_INTERVAL_HOURS", "6"))
    if heartbeat_hours > 0:
        async def _heartbeat_job(context: ContextTypes.DEFAULT_TYPE) -> None:
            logger.info("Running scheduled heartbeat...")
            try:
                from agents.seo_agent.heartbeat import execute_heartbeat
                await execute_heartbeat()
            except Exception:
                # NEVER let the heartbeat crash the bot
                logger.error("Heartbeat failed (non-fatal): %s", traceback.format_exc())
                try:
                    import httpx
                    async with httpx.AsyncClient() as client:
                        await client.post(
                            f"https://api.telegram.org/bot{token}/sendMessage",
                            json={"chat_id": int(os.getenv('TELEGRAM_OWNER_CHAT_ID', '7428463356')),
                                  "text": f"Heartbeat error (bot still running): {traceback.format_exc()[-300:]}"},
                            timeout=10,
                        )
                except Exception:
                    pass

        app.job_queue.run_repeating(
            _heartbeat_job,
            interval=heartbeat_hours * 3600,
            first=600,  # First run 10 minutes after startup
            name="heartbeat",
        )
        logger.info("Heartbeat scheduled every %d hours (first run in 10 minutes)", heartbeat_hours)

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
