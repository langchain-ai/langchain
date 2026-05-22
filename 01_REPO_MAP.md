# LangChain Repository Map

## Top-level Structure
```
langchain/
├── libs/
│   ├── core/                    # langchain-core (v1.4.0) - base abstractions
│   ├── langchain/               # langchain-classic (v1.3.1) - legacy
│   ├── langchain_v1/            # Active langchain package
│   ├── partners/                # Partner integrations
│   │   ├── anthropic/           # Claude integration
│   │   ├── openai/             # OpenAI integration
│   │   ├── deepseek/           # DeepSeek integration
│   │   ├── ollama/             # Ollama integration
│   │   ├── groq/               # Groq integration
│   │   ├── fireworks/          # Fireworks AI
│   │   ├── huggingface/        # HuggingFace
│   │   ├── mistralai/          # Mistral AI
│   │   ├── nomic/              # Nomic
│   │   ├── chroma/             # Chroma vector DB
│   │   ├── exa/                # Exa search
│   │   ├── qdrant/             # Qdrant vector DB
│   │   ├── perplexity/         # Perplexity
│   │   ├── openrouter/         # OpenRouter
│   │   └── xai/                # xAI
│   ├── text-splitters/         # Document chunking
│   ├── standard-tests/          # Shared integration tests
│   └── model-profiles/          # Model configuration
├── .github/
│   └── workflows/              # CI/CD (pr_lint, test, integration_tests, etc.)
├── AGENTS.md                   # Development guidelines
└── README.md
```

## Key Files
- `AGENTS.md` - Global development guidelines (292 lines)
- `libs/core/pyproject.toml` - Core package dependencies
- `libs/langchain_v1/pyproject.toml` - Main langchain package
- `.github/workflows/pr_lint.yml` - Conventional commits validation
- `.github/workflows/_test.yml` - Unit test workflow
- `.github/workflows/_lint.yml` - Linting workflow

## Package Managers & Tools
- **uv** - Fast Python package manager (replaces pip/poetry)
- **ruff** - Linter and formatter
- **mypy** - Static type checker
- **pytest** - Testing framework

## PR Conventions
- Title: `type(scope): description` (lowercase after colon unless proper noun)
- Allowed types: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert, release, hotfix
- Allowed scopes: core, langchain, langchain-classic, model-profiles, standard-tests, text-splitters, docs, + partner packages
- PR description must have `Fixes #issue` at top if closing an issue

## Notable Source Files (Bug-Fix Candidates)
- `libs/langchain/langchain_classic/retrievers/re_phraser.py` - RePhraseQueryRetriever (async stub NotImplementedError)
- `libs/partners/anthropic/langchain_anthropic/chat_models.py` - ChatAnthropic.bind_tools mutation bug