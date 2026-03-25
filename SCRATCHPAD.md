# Scratchpad
[Temporary working notes, drafts, and intermediate reasoning. Not persisted long-term.]

## Current session notes
- OpenRouter model IDs differ from Anthropic native IDs:
  - Haiku: `anthropic/claude-haiku-4.5` (not `claude-haiku-4-5-20251001`)
  - Sonnet: `anthropic/claude-sonnet-4.6`
  - Opus: `anthropic/claude-opus-4.6`
- Anthropic SDK `base_url` override does NOT work with OpenRouter — must use OpenAI SDK
- Supabase REST API requires service role key to list tables (anon key returns 401 on schema)
- Direct Postgres connections blocked from cloud sandboxes — use REST API or dashboard SQL editor
- Railway needs explicit Dockerfile — Nixpacks can't figure out a Python monorepo
- `.dockerignore` must exclude `libs/` (43MB) and `.git/` (448MB) or builds are painfully slow
