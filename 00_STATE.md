# LangChain Repository Analysis - State

## Repository Info
- **Full name**: langchain-ai/langchain
- **Fork**: okwn/langchain (master branch)
- **License**: MIT
- **Archived**: No
- **Stars**: 137,399
- **Forks**: 22,728
- **Open Issues**: 582
- **Default Branch**: master

## Architecture
Python monorepo using `uv` for package management. Multiple independently versioned packages under `libs/`:
- `core/` - langchain-core (v1.4.0)
- `langchain/` - langchain-classic (legacy, v1.3.1)
- `langchain_v1/` - active langchain package
- `partners/` - third-party integrations (openai, anthropic, ollama, deepseek, etc.)
- `text-splitters/`, `standard-tests/`, `model-profiles/`

## Key Technologies
- Python 3.10-3.14
- pydantic >=2.7.4
- langgraph >=1.2.0
- pytest, ruff, mypy for testing/linting

## Development Workflow
- `uv sync --all-groups` - Install all packages
- `make test` - Run unit tests
- `make lint` - Lint code
- `make format` - Format code
- PR titles follow Conventional Commits: `type(scope): description`

## Recent Activity (as of 2026-05-22)
- Recent open PRs about PII middleware, conditional interrupts, HITL improvements
- Bug: RePhraseQueryRetriever._aget_relevant_documents raises NotImplementedError
- Bug: ChatAnthropic.bind_tools mutates caller-provided tool_choice dict