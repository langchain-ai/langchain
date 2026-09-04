---
type: "Reference"
title: "Source Map: Repository File Organization"
description: "Quick reference for locating code by topic, mapping LangChain concepts to their implementation paths across the monorepo including core abstractions, agents, middleware, partners, and configuration files."
tags: [reference, file-organization, monorepo, codebase-map, pathfinding]
verified:
  - by: openwiki/0.5.0
    at: 2026-09-03T15:18:34.589Z
sources:
  - id: openwiki-source-4d1645cb6317345817452838
    resource: repo://.pre-commit-config.yaml
  - id: openwiki-source-1c233fccf5a66b84d0045366
    resource: repo://libs/core/langchain_core/callbacks/manager.py
  - id: openwiki-source-c52037e7b642f7ac5a7642a8
    resource: repo://libs/core/langchain_core/language_models/chat_models.py
  - id: openwiki-source-b32b84365d17276620c41ebc
    resource: repo://libs/core/langchain_core/messages/base.py
  - id: openwiki-source-03d7415879ed05a392edd62d
    resource: repo://libs/core/langchain_core/prompts/base.py
  - id: openwiki-source-a1981e868973f6fd7f71e12e
    resource: repo://libs/core/langchain_core/runnables/base.py
  - id: openwiki-source-4ff475d7b00540f962384251
    resource: repo://libs/core/langchain_core/tools/base.py
  - id: openwiki-source-3486a94e6eb23a78271a5bfb
    resource: repo://libs/core/pyproject.toml
  - id: openwiki-source-a690cf632a02205e3f555be8
    resource: repo://libs/core/tests/unit_tests/test_tools.py
  - id: openwiki-source-71e882e1ac9757ea8e959a7c
    resource: repo://libs/langchain_v1/langchain/agents/factory.py
  - id: openwiki-source-03e8ca0eebe37feda8566793
    resource: repo://libs/langchain_v1/langchain/agents/middleware/types.py
  - id: openwiki-source-ec30ab6256dd50cc670919f6
    resource: repo://libs/langchain_v1/langchain/agents/structured_output.py
  - id: openwiki-source-c479d4fffee5cf62576699e4
    resource: repo://libs/langchain_v1/langchain/chat_models/base.py
  - id: openwiki-source-0a3228970b0eadc4bcadbb5d
    resource: repo://libs/langchain_v1/langchain/mcp/adapter.py
  - id: openwiki-source-ba4876d385d4d18ed4fa0342
    resource: repo://libs/langchain_v1/pyproject.toml
  - id: openwiki-source-94d5218d78dfd52679adc96b
    resource: repo://libs/langchain_v1/tests/unit_tests/test_imports.py
  - id: openwiki-source-49fbcc45434b619b68220bf9
    resource: repo://libs/Makefile
  - id: openwiki-source-77f5d6298c73161b4d4f697e
    resource: repo://libs/model-profiles/langchain_model_profiles/__init__.py
  - id: openwiki-source-738512768ef81ae009b097ac
    resource: repo://libs/partners/openai/langchain_openai/chat_models/base.py
  - id: openwiki-source-bd29e79613d5f366a00068f5
    resource: repo://libs/standard-tests/langchain_tests/base.py
generated: { by: "openwiki/0.5.0", at: "2026-09-03T15:18:34.589Z" }
---

## Overview

This page provides a quick reference for locating code by topic in the LangChain monorepo. The repository is organized as a multi-package workspace with a three-layer architecture: **langchain-core** (base abstractions), **langchain** (orchestration and agents), and **partners** (provider integrations). Use this map to navigate directly to the code responsible for a given concept.

## Concept-to-Path Mapping

| Concept | Primary Path | Purpose |
|---------|--------------|---------|
| **Agent Factory** | `repo://libs/langchain_v1/langchain/agents/factory.py` | Constructs compiled LangGraph state machines for agentic loops with model binding, tool execution, and middleware composition |
| **Agent Middleware** | `repo://libs/langchain_v1/langchain/agents/middleware/` | Pluggable hooks for model calls, tool invocation, and lifecycle events (retries, human-in-loop, redaction, etc.) |
| **Chat Models** | `repo://libs/core/langchain_core/language_models/chat_models.py` | `BaseChatModel` abstract base and unified provider interface for streaming, batching, and token counting |
| **Chat Models (Core Interfaces)** | `repo://libs/core/langchain_core/language_models/` | Base language model classes, fake models for testing, and compatibility bridges |
| **Chat Models (Partner Implementations)** | `repo://libs/partners/*/` (e.g., `openai/`, `anthropic/`, `ollama/`) | Provider-specific implementations: ChatOpenAI, ChatAnthropic, ChatOllama, etc. |
| **Callbacks & Tracing** | `repo://libs/core/langchain_core/callbacks/` | Callback manager, base handlers, streaming output, and LangSmith integration |
| **Messages** | `repo://libs/core/langchain_core/messages/` | Message types (AIMessage, HumanMessage, SystemMessage, ToolMessage) and content blocks |
| **Model Initialization** | `repo://libs/langchain_v1/langchain/chat_models/base.py` | `init_chat_model()` factory for dynamic model discovery and loading by provider:model identifier |
| **Model Profiles** | `repo://libs/model-profiles/langchain_model_profiles/` | Metadata and profiles for LLM behavior, capabilities, and configuration |
| **MCP (Model Context Protocol)** | `repo://libs/langchain_v1/langchain/mcp/` | Adapter, tools, and elicitation for MCP-based model integrations |
| **Output Parsers** | `repo://libs/core/langchain_core/output_parsers/` | Structured output parsing, Pydantic model marshaling, and output validation |
| **Prompts** | `repo://libs/core/langchain_core/prompts/` | Chat and string prompt templates, few-shot examples, and image prompts |
| **Runnables (LCEL)** | `repo://libs/core/langchain_core/runnables/` | Foundational `Runnable[Input, Output]` protocol, composition operators, and control flow (piping, branching, fallback, retry) |
| **Structured Output** | `repo://libs/langchain_v1/langchain/agents/structured_output.py` | Schema definition and response marshaling for typed agent outputs |
| **Tools** | `repo://libs/core/langchain_core/tools/` | `BaseTool` abstraction, tool conversion from functions/Pydantic, and tool rendering |
| **Tests (Core Unit)** | `repo://libs/core/tests/unit_tests/` | Unit tests for abstractions: runnables, messages, prompts, tools, callbacks |
| **Tests (Core Integration)** | `repo://libs/core/tests/integration_tests/` | Integration tests with live providers and external services |
| **Tests (LangChain Unit)** | `repo://libs/langchain_v1/tests/unit_tests/` | Unit tests for agent factory, middleware, chat models, and orchestration |
| **Tests (LangChain Integration)** | `repo://libs/langchain_v1/tests/integration_tests/` | Integration tests for agent execution, tool binding, and provider fallback |
| **Tests (Partner Unit)** | `repo://libs/partners/*/tests/unit_tests/` | Provider-specific unit tests |
| **Tests (Partner Integration)** | `repo://libs/partners/*/tests/integration_tests/` | Provider-specific integration tests |
| **Standard Tests** | `repo://libs/standard-tests/langchain_tests/` | Shared test suites and contracts for component conformance across the ecosystem |
| **Configuration (Core)** | `repo://libs/core/pyproject.toml` | langchain-core package metadata, dependencies (langsmith, httpx, tenacity, pydantic), and build config |
| **Configuration (LangChain)** | `repo://libs/langchain_v1/pyproject.toml` | langchain package metadata, core dependencies, and optional provider groups |
| **Configuration (Repo-Wide)** | `repo:///.pre-commit-config.yaml` | Git hooks for formatting, linting, and validation across all packages |
| **Build System (Libs)** | `repo://libs/Makefile` | Monorepo-level build targets, dependency locking, and cross-package tasks |

## Key Directory Structure

```
/libs/
├── core/                           # langchain-core: Base abstractions (v1.6.1)
│   ├── langchain_core/
│   │   ├── language_models/        # BaseChatModel and language model abstractions
│   │   ├── messages/               # Message types and content blocks
│   │   ├── runnables/              # Runnable protocol and composition
│   │   ├── tools/                  # BaseTool and tool conversion
│   │   ├── prompts/                # Prompt templates and few-shot
│   │   ├── output_parsers/         # Output parsing and validation
│   │   ├── callbacks/              # Callback manager and handlers
│   │   ├── retrievers.py           # Retriever abstraction
│   │   └── ... (other modules)
│   ├── tests/
│   │   ├── unit_tests/
│   │   └── integration_tests/
│   ├── Makefile
│   └── pyproject.toml
│
├── langchain_v1/                   # langchain: Orchestration and agents (v1.4.0)
│   ├── langchain/
│   │   ├── agents/
│   │   │   ├── factory.py          # Agent factory and graph construction
│   │   │   ├── middleware/         # Pluggable middleware system
│   │   │   ├── structured_output.py # Response schema and marshaling
│   │   │   └── _subagent_transformer.py # Sub-agent utilities
│   │   ├── chat_models/
│   │   │   └── base.py             # init_chat_model factory
│   │   ├── mcp/                    # Model Context Protocol adapter
│   │   ├── messages/               # v1-specific message utilities
│   │   ├── embeddings/             # Embedding utilities
│   │   ├── tools/                  # v1-specific tool utilities
│   │   └── rate_limiters/          # Rate limiting implementations
│   ├── tests/
│   │   ├── unit_tests/
│   │   ├── integration_tests/
│   │   ├── benchmarks/
│   │   └── cassettes/              # VCR cassettes for HTTP mocking
│   ├── Makefile
│   └── pyproject.toml
│
├── partners/                       # Provider-specific integrations
│   ├── openai/                     # ChatOpenAI, embeddings, etc.
│   ├── anthropic/                  # ChatAnthropic (Claude)
│   ├── ollama/                     # ChatOllama (local models)
│   ├── groq/                       # ChatGroq
│   ├── mistralai/                  # ChatMistralAI
│   ├── huggingface/                # HuggingFace embeddings and models
│   ├── deepseek/                   # ChatDeepSeek
│   ├── xai/                        # XAI (Grok)
│   ├── perplexity/                 # Perplexity models
│   ├── fireworks/                  # Fireworks inference
│   ├── openrouter/                 # OpenRouter aggregator
│   ├── chroma/                     # Chroma vector store
│   ├── qdrant/                     # Qdrant vector store
│   ├── exa/                        # Exa search
│   ├── nomic/                      # Nomic embeddings
│   └── Makefile
│
├── model-profiles/                 # LLM behavior and capability profiles
│   ├── langchain_model_profiles/
│   ├── Makefile
│   └── pyproject.toml
│
├── standard-tests/                 # Cross-ecosystem test contracts
│   ├── langchain_tests/
│   ├── tests/
│   ├── Makefile
│   └── pyproject.toml
│
├── text-splitters/                 # Text splitting utilities
│
├── Makefile                        # Multi-package build coordination
└── README.md
```

## Common Workflows

### Finding Agent-Related Code
- **Agent construction**: `repo://libs/langchain_v1/langchain/agents/factory.py`
- **Middleware hooks**: `repo://libs/langchain_v1/langchain/agents/middleware/types.py` for type definitions; individual middleware in `repo://libs/langchain_v1/langchain/agents/middleware/` subdirectory
- **Structured output**: `repo://libs/langchain_v1/langchain/agents/structured_output.py`
- **Tests**: `repo://libs/langchain_v1/tests/unit_tests/` and `repo://libs/langchain_v1/tests/integration_tests/`

### Finding LLM Integration Code
- **Provider implementations**: `repo://libs/partners/<provider>/` (e.g., `repo://libs/partners/openai/`)
- **Model discovery**: `repo://libs/langchain_v1/langchain/chat_models/base.py` (init_chat_model)
- **Base interface**: `repo://libs/core/langchain_core/language_models/chat_models.py`
- **Model profiles**: `repo://libs/model-profiles/langchain_model_profiles/`

### Finding Core Abstractions
- **Runnable protocol**: `repo://libs/core/langchain_core/runnables/base.py`
- **Messages**: `repo://libs/core/langchain_core/messages/`
- **Tools**: `repo://libs/core/langchain_core/tools/base.py`
- **Prompts**: `repo://libs/core/langchain_core/prompts/`
- **Callbacks**: `repo://libs/core/langchain_core/callbacks/`

### Finding Tests
- **Core abstractions**: `repo://libs/core/tests/`
- **Agent and orchestration**: `repo://libs/langchain_v1/tests/`
- **Provider-specific**: `repo://libs/partners/<provider>/tests/`
- **Shared test contracts**: `repo://libs/standard-tests/`

### Configuration and Build
- **Lint and format**: `.pre-commit-config.yaml` at repo root
- **Core dependencies**: `repo://libs/core/pyproject.toml`
- **LangChain dependencies**: `repo://libs/langchain_v1/pyproject.toml`
- **Build tasks**: `repo://libs/Makefile` for multi-package commands

## Build and Development Commands

All package directories (`libs/core/`, `libs/langchain_v1/`, `libs/partners/<provider>/`, etc.) include a local `Makefile` with standard targets:

```bash
# Format code (ruff)
make -C <package> format

# Lint code (ruff)
make -C libs/core lint

# Run all tests
make -C libs/langchain_v1 test

# Lock dependencies
make -C libs/core lock

# Check lockfile consistency
make -C libs/core check-lock
```

The root `repo://libs/Makefile` coordinates multi-package operations:

```bash
# Lock all packages at once
make -C libs lock

# Check all lockfiles
make -C libs check-lock
```

## Key Implementation Artifacts

### Agent Factory Graph
The agent construction pipeline in `repo://libs/langchain_v1/langchain/agents/factory.py` builds a LangGraph state machine with these nodes:
- **Entry**: Runs `before_agent` middleware once
- **Loop Entry**: Begins each model iteration, runs `before_model` hooks
- **Model**: Invokes language model with message history
- **After Model**: Runs `after_model` hooks for response processing
- **Tools**: Executes tool calls (if any)
- **Exit**: Runs `after_agent` hooks once at completion

Middleware can inject hooks at model boundaries, tool boundaries, and lifecycle hooks (`before_agent`, `before_model`, `after_model`, `after_tool_call`, `after_agent`).

### Runnable Composition
The Runnable protocol in `repo://libs/core/langchain_core/runnables/base.py` enables declarative chaining via operators:
- **Piping** (`|`): Sequential composition
- **Parallel** (`+`): Parallel execution branches
- **Branching** (`.pipe()` with routing): Conditional execution paths
- **Fallback** (`.with_fallback()`): Error recovery with alternatives
- **Retry** (`.with_retry()`): Automatic retry with backoff

All compositions automatically support `invoke()`, `ainvoke()`, `batch()`, `stream()`, and async variants.

### Message Protocol
The message abstraction in `repo://libs/core/langchain_core/messages/` defines:
- **Message types**: `AIMessage`, `HumanMessage`, `SystemMessage`, `ToolMessage`, `FunctionMessage`
- **Content blocks**: `TextBlock`, `ImageBlock`, `ToolUseBlock`, `ToolResultBlock`, custom blocks
- **Message utilities**: Serialization, merging, role mapping, model-specific translation

### Tool Abstraction
The `BaseTool` in `repo://libs/core/langchain_core/tools/base.py` provides:
- **Tool protocol**: Sync/async invoke, schema generation from docstrings/Pydantic
- **Conversion**: Helper functions to wrap Python functions as tools
- **Rendering**: Format tools for model context as descriptions or structured schemas

## Dependency Flow

```
User Applications
  ├─→ langchain (v1.4.0)
  │    ├─→ langchain-core (v1.6.1)
  │    └─→ LangGraph (state machines)
  │
  ├─→ langchain-core (direct use)
  │
  └─→ Partner Packages (langchain-openai, langchain-anthropic, etc.)
       └─→ Implement langchain-core abstractions
```

**Versioning**: Core is released independently with strict semantic versioning. LangChain and partners pin core versions. Partner packages are released independently per provider.
