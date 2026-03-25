# Plan

## Goal
Build and operate an autonomous SEO agent that grows organic traffic across kitchensdirectory.co.uk, freeroomplanner.com, and the kitchen cost estimator — managed entirely via Telegram.

## Phases
1. **Infrastructure** (DONE) — Supabase database, OpenRouter LLM routing, Railway deployment, Telegram bot
2. **Data sources** (NEXT) — Connect Ahrefs API, Google Search Console, Tavily web search
3. **Content pipeline** — Keyword research → content gap analysis → briefs → published articles
4. **Outreach pipeline** — Prospect discovery → enrichment → scoring → email generation → sending via Resend
5. **Automation** — Scheduled weekly reports, automatic rank tracking, budget alerts via Telegram
6. **Optimisation** — A/B test outreach templates, refine scoring model, expand to new site profiles

## Open Questions
- Which Ahrefs plan is needed for the API access level required?
- Should GSC be connected via service account or OAuth?
- What domain will the kitchen estimator live on?
- Should outreach emails come from a dedicated subdomain to protect main domain reputation?
- What is the target number of backlinks per month?

## Success Criteria
- Telegram bot responds to natural language within 5 seconds for simple queries
- Weekly keyword research produces 20+ actionable opportunities per site
- Content briefs are published-ready with minimal human editing
- Outreach pipeline runs autonomously with human approval only at the send step
- Weekly LLM spend stays under $50 cap
- Rank improvements measurable within 90 days of content publication
