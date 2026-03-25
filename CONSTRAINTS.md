# Constraints

## Hard Limits (never do these)
- Never send unsolicited outreach to .gov.uk or .ac.uk domains
- Never exceed 20 outreach emails per day
- Never re-contact a domain within 90 days
- Never send outreach outside 08:00–17:00 UK time
- Never commit API keys, tokens, or passwords to Git
- Never skip LLM cost logging
- Never bypass the outreach blocklist

## Compliance & Legal
- GDPR applies — outreach emails must include opt-out mechanism
- CAN-SPAM: must identify as commercial, include physical address, honour unsubscribes
- Ahrefs ToS: respect API rate limits, don't redistribute raw data
- Google Search Console ToS: don't share GSC data publicly
- OpenRouter ToS: don't use for prohibited content categories

## Technical Constraints
- **LLM budget**: $50/week maximum, auto-downgrade at 80%
- **Telegram message limit**: 4096 characters per message
- **Railway**: worker service, no persistent disk (use Supabase for all state)
- **OpenRouter rate limits**: vary by model and plan
- **Supabase free tier**: 500MB database, 2GB bandwidth, 50MB file storage
- **Docker image size**: keep under 500MB by excluding `libs/` and `.git/`

## Scope Constraints
- Agent manages SEO only — no paid advertising, social media, or email marketing
- Three sites only: kitchensdirectory, freeroomplanner, kitchen_estimator
- English language content only
- UK primary market, US/Canada secondary
- No automated publishing — agent generates drafts, human reviews and publishes
