# Scraper Agent (Stephen Scraper Bot)

Telegram bot that finds kitchen and bathroom companies using Tavily search + Firecrawl extraction and adds them to the shared Supabase CRM.

## Deployment

The scraper bot is a **separate service** from the SEO bot (Ralf). It has its own Dockerfile and Railway config:

- `Dockerfile.scraper` — Docker image for the scraper bot
- `railway.scraper.toml` — Railway service config (points to `Dockerfile.scraper`)

When creating a Railway service for the scraper bot, set the **config file path** to `railway.scraper.toml` in the service settings.

Do **not** use the default `Dockerfile` / `railway.toml` — those start the SEO bot.

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `SCRAPER_TELEGRAM_BOT_TOKEN` | Yes | Telegram bot token for Stephen Scraper Bot |
| `TELEGRAM_OWNER_CHAT_ID` | Yes | Chat ID for scheduled batch notifications |
| `TAVILY_API_KEY` | Yes | Tavily search API key |
| `FIRECRAWL_API_KEY` | No | Firecrawl extraction key (falls back to regex) |
| `SUPABASE_URL` | Yes | Shared Supabase project URL |
| `SUPABASE_SERVICE_KEY` | Yes | Supabase service role key |
| `SCRAPER_DAILY_TARGET` | No | Daily target (default: 50) |
| `SCRAPE_INTERVAL_HOURS` | No | Batch interval in hours (default: 6) |

## Commands

| Command | Description |
|---|---|
| `/scrape <country> [category]` | Run a scrape (UK, US, CA) |
| `/scrape_all` | Scrape all countries |
| `/batch` | Run a daily batch (targets 50/day) |
| `/status` | CRM stats |
| `/today` | Today's scraping progress |
| `/help` | Show commands |

## Running locally

```bash
export SCRAPER_TELEGRAM_BOT_TOKEN=your_token
export TAVILY_API_KEY=your_key
export SUPABASE_URL=your_url
export SUPABASE_SERVICE_KEY=your_key
python -m agents.scraper_agent.telegram_bot
```
