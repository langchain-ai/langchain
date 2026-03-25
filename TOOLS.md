# Tools

## Available Tools
| Tool | Purpose | When to Use |
|------|---------|-------------|
| OpenRouter API | LLM inference (Claude Haiku/Sonnet/Opus) | Every agent task that needs text generation/classification |
| Supabase REST API | Database CRUD for all SEO data | Persisting keywords, prospects, briefs, emails, cost logs |
| Supabase Postgres | Direct SQL access for migrations | Initial table creation, schema changes |
| Ahrefs API | Keyword data, backlink analysis, DR scores | Keyword research, content gap, prospect discovery |
| Google Search Console API | Rank tracking, impressions, clicks | Rank reports, weekly reports |
| Tavily API | Web search for page content and contacts | Prospect enrichment, page summarisation |
| Resend API | Transactional email sending | Outreach email delivery |
| Telegram Bot API | User interface via long polling | All user interactions |
| GitHub CLI (`gh`) | Repo management, pushing code | Deployment pipeline |
| Railway CLI | Deployment management | Service config, env vars |

## Tool Notes
- **OpenRouter**: rate limits vary by model; Haiku is fastest. Responses cached in `llm_output_cache` table.
- **Supabase**: service role key required for table listing. Anon key only for authenticated user operations.
- **Ahrefs**: API key not yet configured. Set `AHREFS_MOCK=true` to use mock data during development.
- **GSC**: requires a service account JSON file. Set `GSC_MOCK=true` to skip.
- **Tavily**: set `TAVILY_MOCK=true` to use mock responses. Rate limit: depends on plan.
- **Resend**: outreach emails must respect warm-up schedule and daily send limits in `config.py`.
- **Telegram**: 4096 character limit per message. Split long responses. Typing indicator sent before LLM calls.
