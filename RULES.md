# Rules

## Hard Rules (never violate)
- Never send outreach emails without explicit user confirmation (`--send` flag or Telegram approval)
- Never exceed MAX_WEEKLY_SPEND_USD ($50) — downgrade models at 80%
- Never contact domains on the `seo_outreach_blocklist` table
- Never contact `.gov.uk` or `.ac.uk` domains with commercial outreach
- Never send outreach outside 08:00–17:00 UK time
- Never re-contact a domain within 90 days (MIN_DAYS_BETWEEN_DOMAIN_CONTACTS)
- Never send more than 20 outreach emails per day (MAX_DAILY_OUTREACH_EMAILS)
- Never commit secrets to Git — all API keys go in `.env` (gitignored) or Railway env vars
- Never skip the LLM cost logger — every `call_llm` invocation must log to `llm_cost_log`
- Minimum outreach score of 35 before any email is generated

## Soft Rules (follow unless there's good reason not to)
- Default to kitchensdirectory when site is ambiguous
- Use Haiku for classification/extraction tasks, Sonnet for writing, Opus only for tier-1 outreach
- Check the `llm_output_cache` before making an LLM call for cacheable tasks
- Follow the warm-up schedule for outreach: week 1 = 5/day, week 2 = 10/day, week 3+ = 20/day
- Pause all outreach sends if bounce rate exceeds 3%
- Keep Telegram messages under 4000 characters (split if longer)
- Use per-user conversation history (20 messages) for NL context
