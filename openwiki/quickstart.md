---
type: "Getting Started"
title: "LangChain Repository Quick Start"
description: "Entry point for engineers: orient to the monorepo structure, run first tests, understand what to edit for common tasks, and route to major development areas."
tags: [quickstart, getting-started, monorepo, setup, development, first-steps, cli-reference]
verified:
  - by: openwiki/0.5.0
    at: 2026-09-03T15:18:34.589Z
sources:
  - id: openwiki-source-4d1645cb6317345817452838
    resource: repo://.pre-commit-config.yaml
  - id: openwiki-source-8e384445ccfbaf00747b3e18
    resource: repo://libs/core/langchain_core/__init__.py
  - id: openwiki-source-96c73d2c05223b0b46abdbe9
    resource: repo://libs/core/langchain_core/callbacks/__init__.py
  - id: openwiki-source-c52037e7b642f7ac5a7642a8
    resource: repo://libs/core/langchain_core/language_models/chat_models.py
  - id: openwiki-source-0f6ea1dd09fb4675ff4112b1
    resource: repo://libs/core/langchain_core/messages/__init__.py
  - id: openwiki-source-e7908d069731cffec228727e
    resource: repo://libs/core/langchain_core/output_parsers/__init__.py
  - id: openwiki-source-c46a1d181ab64e61460c84c6
    resource: repo://libs/core/langchain_core/prompts/__init__.py
  - id: openwiki-source-65071982a626569c8820a34b
    resource: repo://libs/core/langchain_core/runnables/__init__.py
  - id: openwiki-source-7c4bed110359f4f5c7847c8b
    resource: repo://libs/core/langchain_core/tools/__init__.py
  - id: openwiki-source-8f1875229ad4a704c8e20a06
    resource: repo://libs/core/Makefile
  - id: openwiki-source-3486a94e6eb23a78271a5bfb
    resource: repo://libs/core/pyproject.toml
  - id: openwiki-source-47db752fe27393d5d4825827
    resource: repo://libs/langchain_v1/langchain/__init__.py
  - id: openwiki-source-71e882e1ac9757ea8e959a7c
    resource: repo://libs/langchain_v1/langchain/agents/factory.py
  - id: openwiki-source-07e634f5cd5f00c636010306
    resource: repo://libs/langchain_v1/langchain/agents/middleware/__init__.py
  - id: openwiki-source-b09b1477098d69af6abaa5b4
    resource: repo://libs/langchain_v1/langchain/chat_models/__init__.py
  - id: openwiki-source-49fbcc45434b619b68220bf9
    resource: repo://libs/Makefile
  - id: openwiki-source-1e66a9da38565f8901e651f4
    resource: repo://libs/partners/openai/langchain_openai/__init__.py
  - id: openwiki-source-48ce5ee900993294d349b4e8
    resource: repo://libs/standard-tests/langchain_tests/__init__.py
generated: { by: "openwiki/0.5.0", at: "2026-09-03T15:18:34.589Z" }
---

## Welcome to LangChain Development

LangChain is the agent engineering platform—a framework for building LLM-powered applications with composable abstractions, provider integrations, and orchestration primitives. This page guides you through the monorepo structure, essential setup, common dev tasks, and routing to deeper documentation.

**New to the repo?** Start with [Installation & Setup](#installation--setup), then jump to [Quick Navigation](#quick-navigation-to-major-areas) to find what you need to work on.

## Monorepo Overview

LangChain is organized as a **three-layer architecture** in `/libs/`:

```
/libs/
├── core/              # langchain-core: Base abstractions (Runnable, BaseChatModel, tools, prompts, messages)
├── langchain_v1/      # langchain: Agent orchestration, factory, middleware
├── partners/          # Provider-specific integrations (OpenAI, Anthropic, Ollama, etc.)
├── standard-tests/    # Shared test suites for component conformance
├── text-splitters/    # Text splitting utilities
├── model-profiles/    # LLM metadata and capability profiles
└── Makefile           # Monorepo-level build targets
```

### When to Edit Each Layer

| Layer | Edit when you are... | Key files |
|-------|----------------------|-----------|
| **core** | Adding or modifying base abstractions, core interfaces (Runnable, BaseChatModel, messages, tools, prompts), or callbacks. | `libs/core/langchain_core/` |
| **langchain_v1** | Building agent factory features, middleware, model initialization, chat model selection, or high-level orchestration. | `libs/langchain_v1/langchain/agents/`, `libs/langchain_v1/langchain/chat_models/` |
| **partners/{name}** | Adding a new LLM provider (OpenAI, Anthropic, etc.), model-specific features, or provider integrations. | `libs/partners/{provider}/` |

## Installation & Setup

### Clone the Repository

```bash
git clone https://github.com/langchain-ai/langchain.git
cd langchain
```

### Install Dependencies with `uv`

The monorepo uses `uv` for fast, deterministic dependency resolution. Install it once:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via Homebrew
brew install uv
```

Then sync all dependencies in your package:

```bash
# From any libs/ subdirectory, install all groups (test, lint, type, dev)
uv sync --all-groups

# Or install only what you need
uv sync --group test     # For running tests
uv sync --group lint     # For ruff/mypy
```

### Pre-Commit Hooks

Install git hooks to enforce code quality automatically:

```bash
pre-commit install

# To manually run all hooks
pre-commit run --all-files

# To run a specific hook
pre-commit run ruff --all-files
```

Pre-commit hooks run:
- YAML/TOML syntax validation
- Text normalization and trailing whitespace fixes
- **Per-package formatting and linting** (ruff, mypy)
- **Version consistency checks** across `pyproject.toml` files

## Common Development Tasks

### Run Unit Tests

```bash
# From any package directory (libs/core, libs/langchain_v1, etc.)
make test

# Run a specific test file
make test TEST_FILE=tests/unit_tests/agents/test_factory.py

# Run tests in watch mode (auto-rerun on file changes)
make test_watch

# Run extended tests (marked @pytest.mark.requires)
make extended_tests
```

**Key details:**
- Tests run with **socket restrictions** (`--disable-socket`) to prevent accidental network calls
- Tests run **in parallel** via pytest-xdist (`-n auto`)
- LangSmith tracing variables are unset to keep tests isolated
- Typical test path mirrors source: `langchain_core/runnables/base.py` → `tests/unit_tests/runnables/test_base.py`

### Format and Lint

```bash
# Format all Python files (ruff)
make format

# Check linting issues (ruff, mypy)
make lint

# Type checking only (mypy)
make type

# Format only changed files (git diff against main)
make format_diff
```

**Tools used:**
- **ruff**: Fast Python linter and formatter (replaces black, isort, flake8)
- **mypy**: Static type checker
- Both are run via `uv run --group lint`

### Full Local Validation

Run this before pushing a PR:

```bash
# From your package directory
make format && make lint && make test
```

Or in one line:

```bash
cd libs/core && make format lint test
```

## Quick Navigation to Major Areas

Use the table below to route to detailed documentation:

| Task | Start Here | Key Concepts |
|------|-----------|--------------|
| **Build an agent** | [Agent Factory](/openwiki/agent-factory.md) | create_agent, AgentState, middleware composition, graph execution |
| **Add a new LLM provider** | [Adding a Chat Model Provider](/openwiki/partner-pattern.md) | ChatModel impl, message conversion, provider registration, standard tests |
| **Understand the architecture** | [Architecture Overview](/openwiki/architecture.md) | Three-layer design, dependency flow, core vs. orchestration vs. partners |
| **Work with chat models** | [Chat Model Interface](/openwiki/chat-models.md) | BaseChatModel protocol, streaming, tool binding, structured output |
| **Initialize models dynamically** | [Model Initialization](/openwiki/model-initialization.md) | init_chat_model factory, provider:model syntax, fallback chains |
| **Compose components (chains, pipelines)** | [Runnables & Composability](/openwiki/runnables.md), [Composability](/openwiki/composability.md) | Runnable protocol, \| operator, branching, retry, fallback |
| **Work with tools** | [Tools](/openwiki/tools.md) | BaseTool, schema generation, tool calling, result handling |
| **Stream responses** | [Streaming](/openwiki/streaming.md) | Token-by-token output, streaming across components |
| **Enforce response formats** | [Structured Output](/openwiki/structured-output.md) | JSON schemas, response validation, typed outputs |
| **Write middleware** | [Agent Middleware](/openwiki/middleware.md) | Middleware types, composition, custom hooks |
| **Trace agent execution** | [Agent Execution Flow](/openwiki/agent-execution.md) | Runtime lifecycle, loop control, state transitions |
| **Add observability** | [Callbacks & Tracing](/openwiki/callbacks.md) | Callback manager, LangSmith integration, logging |
| **Write unit/integration tests** | [Unit Testing](/openwiki/unit-tests.md), [Integration Testing](/openwiki/integration-tests.md) | Test structure, fixtures, mocking, VCR cassettes |
| **Work with prompts** | [Prompts](/openwiki/prompts.md) | Templates, few-shot, variables, image handling |
| **Understand message types** | [Messages](/openwiki/messages.md) | AIMessage, ToolMessage, content blocks, provider conversion |
| **Use Model Context Protocol** | [MCP Integration](/openwiki/mcp-integration.md) | MCP servers, tool adapters, elicitation |
| **Reference all file paths** | [Source Map](/openwiki/source-map.md) | Concept-to-path lookup table, directory structure |
| **Check CI/CD workflows** | [CI/CD Workflows](/openwiki/ci-workflows.md) | GitHub Actions, testing, linting, release process |
| **Development command reference** | [Dev Commands](/openwiki/dev-commands.md) | Detailed make targets, uv syntax, env setup |

## Repository Structure at a Glance

### Root Level

```
/
├── .github/              # GitHub Actions workflows (CI/CD)
├── .pre-commit-config.yaml # Pre-commit hooks definition
├── .vscode/              # VS Code settings
├── libs/                 # Main monorepo workspace
├── AGENTS.md             # Agent-focused documentation
├── CLAUDE.md             # Contributing guide (READ THIS BEFORE PR)
└── README.md             # Top-level project overview
```

### Inside `/libs/`

**core/** — Base abstractions (langchain-core package)
```
core/
├── langchain_core/
│   ├── language_models/  # BaseChatModel and language model contracts
│   ├── messages/         # Message types and content blocks
│   ├── runnables/        # Runnable protocol and operators
│   ├── tools/            # BaseTool and tool utilities
│   ├── prompts/          # Prompt templates and few-shot
│   ├── callbacks/        # Callback manager and handlers
│   └── output_parsers/   # Output parsing and validation
├── tests/unit_tests/     # Unit tests (no network)
├── tests/integration_tests/ # Integration tests (live APIs)
├── Makefile              # Build targets (test, lint, format)
└── pyproject.toml        # Package deps and metadata
```

**langchain_v1/** — Agent orchestration (langchain package)
```
langchain_v1/
├── langchain/
│   ├── agents/
│   │   ├── factory.py    # create_agent function
│   │   ├── middleware/   # Pluggable middleware hooks
│   │   └── structured_output.py # Response schema
│   ├── chat_models/
│   │   └── base.py       # init_chat_model factory
│   ├── mcp/              # Model Context Protocol
│   └── ...
├── tests/unit_tests/
├── tests/integration_tests/
├── tests/cassettes/      # VCR cassettes for HTTP mocking
├── Makefile
└── pyproject.toml
```

**partners/** — Provider integrations
```
partners/
├── openai/               # ChatOpenAI, embeddings
├── anthropic/            # ChatAnthropic (Claude)
├── ollama/               # ChatOllama (local models)
├── groq/                 # ChatGroq
├── mistralai/            # ChatMistralAI
├── huggingface/          # HuggingFace models/embeddings
├── deepseek/             # ChatDeepSeek
└── ... (20+ more providers)
```

Each partner has the same structure:
```
provider/
├── langchain_{provider}/
│   ├── __init__.py       # Exports ChatModel class
│   ├── chat_models/
│   │   └── base.py       # ChatModel implementation
│   └── data/             # Model profiles
├── tests/
│   ├── unit_tests/       # Standard tests + custom
│   └── integration_tests/
├── pyproject.toml
├── Makefile
└── uv.lock
```

## Your First PR: A Workflow

### 1. Pick a Task

Decide what you want to work on using the [Quick Navigation](#quick-navigation-to-major-areas) table above. For first-time contributors:
- **Easy**: Add a test, fix a type error, improve documentation
- **Medium**: Add a new middleware hook, extend a tool interface
- **Hard**: Add a new provider integration (follow [Adding a Chat Model Provider](/openwiki/partner-pattern.md))

### 2. Read the Contributing Guide

Before coding, read:
- **[CLAUDE.md](repo://CLAUDE.md)** — Conventions, style, and PR expectations
- **Relevant wiki page** — Deep context on your area (see table above)

### 3. Set Up Your Package

```bash
cd libs/{core|langchain_v1|partners/provider}
uv sync --all-groups
pre-commit install
```

### 4. Make Your Changes

Follow the style and patterns you see in the codebase. Use type hints; write tests alongside code.

### 5. Run Local Checks

```bash
make format lint test
```

All checks must pass before pushing.

### 6. Commit and Push

```bash
git add .
git commit -m "Brief description of change"
git push origin your-branch
```

Pre-commit hooks will run automatically. If they fail, fix and commit again.

### 7. Open a Pull Request

Link the PR to any relevant issue and reference the wiki pages you read in the description. The LangChain team will review and provide feedback.

## Key Files to Know

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Contributing guide, style, and conventions |
| `libs/Makefile` | Monorepo-level make targets (lock, check-lock) |
| `libs/{core,langchain_v1,partners/*/Makefile` | Per-package test, lint, format targets |
| `.pre-commit-config.yaml` | Git hooks for code quality |
| `pyproject.toml` (per-package) | Package metadata, dependencies, build config |

## Troubleshooting

### Tests Fail with Socket Errors
Tests run with socket restrictions by default. If you need network access:
- Write an integration test in `tests/integration_tests/` (see [Integration Testing](/openwiki/integration-tests.md))
- Or disable socket restrictions locally: `uv run --group test pytest --disable-socket=false ...`

### Import Errors or Version Mismatches
Regenerate lockfiles:
```bash
cd libs
make lock
```

Or in a single package:
```bash
cd libs/core
uv lock
```

### Type Checking Fails
Run mypy to see detailed errors:
```bash
make type
```

Check the [Chat Models](/openwiki/chat-models.md) or [Runnables](/openwiki/runnables.md) pages for type signature patterns.

### Pre-Commit Hooks Block Commit
Pre-commit will auto-fix formatting and some issues. Re-stage and commit:
```bash
git add .
git commit -m "..."  # Try again
```

If linting still fails, run `make lint` to see details and fix manually.

## Quick Command Reference

```bash
# Setup
uv sync --all-groups          # Install all dependencies
pre-commit install            # Setup git hooks

# Testing
make test                      # Run unit tests
make test TEST_FILE=path/     # Run specific test file
make test_watch               # Watch mode (auto-rerun)
make integration_tests        # Run integration tests

# Code Quality
make format                   # Format code (ruff)
make lint                     # Check linting (ruff, mypy)
make type                     # Type check only (mypy)

# Lockfile Management
cd libs && make lock          # Regenerate all lockfiles
cd libs && make check-lock    # Verify lockfiles are up-to-date

# All Before PR
make format && make lint && make test
```

## Next Steps

1. **Read [CLAUDE.md](repo://CLAUDE.md)** for contributing conventions
2. **Pick a wiki page** from [Quick Navigation](#quick-navigation-to-major-areas) matching your task
3. **Clone, setup, and make your first change**
4. **Run `make format lint test`** to validate locally
5. **Open a PR** and engage with the team

Welcome to LangChain! 🚀
