# Memory

## Short-Term (current session context)
- Telegram bot deployed with natural language conversation support
- All 12 Supabase tables created and verified
- OpenRouter API key configured and tested (Haiku, Sonnet, Opus routing works)
- Railway deployment successful with Docker build

## Long-Term (persistent facts and decisions)
- Supabase project ref: `ewkrubluzctsfnkxnmsj`
- Supabase URL: `https://ewkrubluzctsfnkxnmsj.supabase.co`
- Database: `postgresql://postgres:***@db.ewkrubluzctsfnkxnmsj.supabase.co:5432/postgres`
- LLM provider: OpenRouter (not direct Anthropic API)
- Model routing: Haiku for classification/extraction, Sonnet for writing/analysis, Opus for tier-1 outreach
- Weekly LLM budget cap: $50 USD
- Budget >80% triggers automatic model downgrading
- Telegram bot: @RalfSEObot (token in Railway env vars)
- GitHub repo: benshevlane/langchain (fork of langchain-ai/langchain)
- Deployment: Railway (Docker, worker service, not web)
- Agent skills: Supabase Postgres Best Practices installed in `.agents/skills/`

## Key Decisions
- **OpenRouter over direct Anthropic** — allows model flexibility and unified billing
- **Provider-agnostic model tiers** — internal haiku/sonnet/opus tiers mapped to provider-specific IDs
- **Telegram over Slack/Discord** — owner preference, mobile-first
- **Natural language routing via Haiku** — cheapest model handles intent classification, heavier models do the actual work
- **Docker deployment** — only copies `agents/` directory, excludes 43MB `libs/` monorepo to keep builds fast
- **exec_sql RPC function** — allows `supabase_tools.py` to create tables dynamically without dashboard access
