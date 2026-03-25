# Conventions

## Naming
- Python files: `snake_case.py`
- Agent nodes: `run_<node_name>` functions in `agents/seo_agent/nodes/`
- Supabase tables: `seo_` prefix for SEO tables, `llm_` prefix for cost/cache tables
- Environment variables: `UPPER_SNAKE_CASE`
- Telegram commands: `/snake_case`

## Code Style
- Python 3.12+, type hints on all public functions
- `from __future__ import annotations` at top of every module
- Docstrings on all public functions (Google style)
- `noqa` comments only where pylint/ruff flags are intentionally suppressed
- Use `logging` module, never `print()` in agent code

## Commit Messages
- Format: `type: short description`
- Types: `feat`, `fix`, `refactor`, `docs`, `chore`, `test`
- Body: bullet points explaining what changed and why
- Reference table names, file paths, or task types when relevant

## Folder Structure
```
langchain/
├── agents/seo_agent/           # All agent code lives here
│   ├── nodes/                  # LangGraph node implementations
│   ├── tools/                  # Supabase, Ahrefs, GSC, LLM router, web search
│   ├── prompts/                # Prompt templates (.txt files)
│   ├── agent.py                # Graph definition (StateGraph)
│   ├── config.py               # Site profiles, token budgets, safety limits
│   ├── state.py                # SEOAgentState TypedDict
│   ├── run.py                  # CLI entrypoint (Click)
│   └── telegram_bot.py         # Telegram bot (python-telegram-bot)
├── migrations/                 # SQL migration files
├── .agents/skills/             # Supabase agent skills
├── libs/                       # Upstream LangChain monorepo (not deployed)
└── tests/seo_agent/            # Agent test suite
```
