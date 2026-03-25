# Architecture

## System Overview
A LangGraph-based SEO agent running as a Railway worker process, accessed via Telegram. The agent orchestrates multiple SEO tasks through a state machine graph, with all data persisted to Supabase and LLM calls routed through OpenRouter.

## Components
| Component | Purpose | Location |
|-----------|---------|----------|
| LangGraph State Machine | Orchestrates SEO tasks as a directed graph | `agents/seo_agent/agent.py` |
| LLM Router | Model selection, cost tracking, output caching | `agents/seo_agent/tools/llm_router.py` |
| Supabase Tools | Database CRUD, table management, cost logging | `agents/seo_agent/tools/supabase_tools.py` |
| Ahrefs Tools | Keyword data, backlink analysis, DR scores | `agents/seo_agent/tools/ahrefs_tools.py` |
| GSC Tools | Google Search Console rank data | `agents/seo_agent/tools/gsc_tools.py` |
| Web Search Tools | Tavily web search for prospect enrichment | `agents/seo_agent/tools/web_search_tools.py` |
| Telegram Bot | Natural language + slash command interface | `agents/seo_agent/telegram_bot.py` |
| CLI | Command-line interface for direct invocation | `agents/seo_agent/run.py` |
| Config | Site profiles, budgets, safety limits | `agents/seo_agent/config.py` |

## Data Flow
```
User (Telegram) → NL Router (Haiku) → Task Classification
                                          ↓
                                    LangGraph State Machine
                                          ↓
                         ┌─────────────────┼─────────────────┐
                         ↓                 ↓                 ↓
                   Ahrefs API        OpenRouter/Claude    Tavily Search
                         ↓                 ↓                 ↓
                         └─────────────────┼─────────────────┘
                                          ↓
                                    Supabase (persist)
                                          ↓
                                    Format results
                                          ↓
                                    Telegram response
```

## Key Dependencies
- `langgraph` — state machine graph execution
- `langchain-core` — base abstractions
- `openai` — OpenRouter API client (OpenAI-compatible format)
- `anthropic` — fallback direct Anthropic API client
- `supabase` — Supabase Python client
- `python-telegram-bot` — Telegram bot framework
- `click` — CLI framework
- `tavily-python` — web search for prospect enrichment
- `resend` — transactional email sending for outreach
- `httpx` + `tenacity` — HTTP client with retry logic

## Decisions & Trade-offs
- **OpenRouter over direct Anthropic**: single billing, model flexibility, but adds a proxy hop
- **OpenAI SDK for OpenRouter**: Anthropic SDK `base_url` override doesn't work with OpenRouter's chat completions format
- **Provider-agnostic tier system**: internal haiku/sonnet/opus names decoupled from provider-specific model IDs — easy to swap providers
- **Long polling over webhooks**: simpler deployment (no public URL needed), acceptable latency for this use case
- **Thread pool for agent tasks**: LangGraph is synchronous, Telegram bot is async — `run_in_executor` bridges the gap
- **Per-user conversation history in memory**: simple deque, lost on restart — acceptable for single-user bot
- **Docker over Nixpacks**: monorepo confused Nixpacks, Dockerfile gives precise control over what gets deployed
