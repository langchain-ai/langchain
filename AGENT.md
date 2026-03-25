# Agent Overview

## Purpose
Ralf is an autonomous SEO agent that manages search engine optimisation across a portfolio of three websites:
- **kitchensdirectory.co.uk** — an independent editorial directory of UK handmade kitchen makers
- **freeroomplanner.com** — a free browser-based room planning tool for homeowners
- **kitchen_estimator** — a kitchen renovation cost estimator (domain TBD)

Ralf handles keyword research, content gap analysis, content briefs and writing, backlink prospecting, outreach email generation, rank tracking, and weekly reporting — all orchestrated via a LangGraph state machine with cost-controlled LLM routing.

## Capabilities
- Keyword research and content gap analysis (via Ahrefs API + LLM)
- Content brief generation and long-form content writing
- Backlink prospect discovery, enrichment, scoring, and outreach email generation
- Rank tracking via Google Search Console
- LLM cost tracking and automatic model downgrading at 80% budget
- Output caching to avoid redundant LLM calls
- Natural language conversation via Telegram bot (@RalfSEObot)
- All data persisted to Supabase (12 tables)

## Invocation
- **Telegram**: message @RalfSEObot in natural language or use slash commands
- **CLI**: `python -m agents.seo_agent.run <command> [options]`
- **Deployed on**: Railway (Docker container, long-polling Telegram bot)

## Owner
Ben Shevlane (benshevlane@gmail.com)
