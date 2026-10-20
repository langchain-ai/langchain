---
type: "Reference"
title: "Source Map: Repository File Organization"
description: "Quick reference for locating code by topic, mapping LangChain concepts to their implementation paths across the monorepo including core abstractions, agents, middleware, partners, and configuration files."
tags: [reference, file-organization, monorepo, codebase-map, pathfinding]
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
  - id: openwiki-source-b65307f1d05bd74532af2eee
    resource: repo://libs/partners/openai/langchain_openai/decisions/base.py
  - id: openwiki-source-bd29e79613d5f366a00068f5
    resource: repo://libs/standard-tests/langchain_tests/base.py
verified:
  - by: openwiki/0.5.0
    at: 2026-10-20T08:25:57.412Z
generated: { by: "openwiki/0.5.0", at: "2026-10-20T08:25:57.412Z" }
---

## Overview

This page provides a quick reference for locating code by topic in the LangChain monorepo. The repository is organized as a multi-package workspace with a three-layer architecture:

- **langchain-core** (v1.6.12): Base abstractions for messages, runnables, tools, prompts, language models, and callbacks
- **langchain** (v1.4.6): Orchestration, agent factory, middleware composition, and model initialization
- **langchain-classic** (v1.0.8): Legacy chains, memory, retrieval abstractions, and backward compatibility
- **partners** (`langchain-*` packages): Provider-specific implementations (OpenAI, Anthropic, Ollama, etc.)
- **text-splitters** (v1.x): Text splitting and chunking utilities
- **model-profiles** (v0.0.6): LLM metadata and capability profiles
- **standard-tests** (shared): Ecosystem test contracts and conformance suites

Use this map to navigate directly to the code responsible for a given concept, understand architectural layering, and locate integration points.

## Concept-to-Path Mapping

| Concept | Primary Path | Purpose |
|---------|--------------|---------|
| **Agent Factory** | `repo://libs/langchain_v1/langchain/agents/factory.py` | Constructs compiled LangGraph state machines for agentic loops with model binding, tool execution, and middleware composition |
| **Agent Middleware** | `repo://libs/langchain_v1/langchain/agents/middleware/` | Pluggable hooks for model calls, tool invocation, and lifecycle events (retries, human-in-loop, redaction, etc.); 20+ middleware implementations |
| **Caches & Prompt Caching** | `repo://libs/core/langchain_core/caches.py` | Optional caching layer for LLM results (distinct from provider prompt caching); in-memory and custom cache backends |
| **Chat Models (Core)** | `repo://libs/core/langchain_core/language_models/chat_models.py` | `BaseChatModel` abstract base and unified provider interface for streaming, batching, token counting, and structured outputs |
| **Chat Models (Discovery)** | `repo://libs/langchain_v1/langchain/chat_models/base.py` | `init_chat_model()` factory for dynamic model discovery and loading by provider:model identifier |
| **Chat Models (Implementations)** | `repo://libs/partners/*/langchain_*/chat_models/` (e.g., `openai/`) | Provider-specific implementations: ChatOpenAI, ChatAnthropic, ChatOllama, ChatGroq, etc. |
| **Callbacks & Tracing** | `repo://libs/core/langchain_core/callbacks/` | Callback manager, base handlers, streaming output, LangSmith integration, and event instrumentation |
| **Decisions (OpenAI)** | `repo://libs/partners/openai/langchain_openai/decisions/` | OpenAI structured outputs API integration for deterministic response marshaling and routing |
| **Document Loaders** | `repo://libs/core/langchain_core/document_loaders/` | Base loaders for ingesting documents from files, URLs, LangSmith, and blob storage |
| **Documents & Metadata** | `repo://libs/core/langchain_core/documents/base.py` | Document type, compression, and transformation utilities |
| **Embeddings** | `repo://libs/core/langchain_core/embeddings/embeddings.py` | Base embedding interface and fake embeddings for testing |
| **Example Selectors** | `repo://libs/core/langchain_core/example_selectors/` | Selection strategies for few-shot examples (LLM, max marginal relevance, etc.) |
| **Messages** | `repo://libs/core/langchain_core/messages/` | Message types (AIMessage, HumanMessage, SystemMessage, ToolMessage) and content blocks (text, images, tool use, etc.) |
| **Model Profiles** | `repo://libs/model-profiles/langchain_model_profiles/` | Metadata registry and CLI tools for LLM behavior, capabilities, and configuration |
| **MCP (Model Context Protocol)** | `repo://libs/langchain_v1/langchain/mcp/` | Adapter and tools for MCP-based model integrations and protocol compatibility |
| **Output Parsers** | `repo://libs/core/langchain_core/output_parsers/` | Structured output parsing (JSON, XML, Pydantic, OpenAI tools), format instructions, and validation |
| **Prompts** | `repo://libs/core/langchain_core/prompts/` | Chat and string prompt templates, few-shot examples, image prompts, and prompt composition |
| **Rate Limiters** | `repo://libs/core/langchain_core/rate_limiters.py` | Base rate limiter interface and in-memory rate limiting implementations |
| **Retrievers** | `repo://libs/core/langchain_core/retrievers.py` | Retriever abstraction for document retrieval, vector store integration, and composition |
| **Runnables (LCEL)** | `repo://libs/core/langchain_core/runnables/` | Foundational `Runnable[Input, Output]` protocol, composition operators (piping, branching, fallback, retry), and execution utilities |
| **Structured Output** | `repo://libs/langchain_v1/langchain/agents/structured_output.py` | Schema definition and response marshaling for typed agent outputs (Pydantic, dataclass, TypedDict, JSON schema) |
| **Tools** | `repo://libs/core/langchain_core/tools/` | `BaseTool` abstraction, tool conversion from functions/Pydantic, schema generation, and tool rendering |
| **Vector Stores** | `repo://libs/core/langchain_core/vectorstores/base.py` | Base vector store interface with in-memory implementation; persistence and async support |
| **Legacy Chains** | `repo://libs/langchain/langchain_classic/chains/` | Legacy chain abstractions (LLMChain, SequentialChain, etc.) for backward compatibility |
| **Legacy Memory** | `repo://libs/langchain/langchain_classic/memory/` | Conversation memory implementations (BufferMemory, EntityMemory, summary-based) |
| **Text Splitters** | `repo://libs/text-splitters/langchain_text_splitters/` | Character, token, semantic, and language-specific text chunking algorithms |
| **Tests (Core Unit)** | `repo://libs/core/tests/unit_tests/` | Unit tests for abstractions: runnables, messages, prompts, tools, callbacks, output parsers |
| **Tests (Core Integration)** | `repo://libs/core/tests/integration_tests/` | Integration tests with live providers and external services |
| **Tests (LangChain Unit)** | `repo://libs/langchain_v1/tests/unit_tests/` | Unit tests for agent factory, middleware, chat models, orchestration, and initialization |
| **Tests (LangChain Integration)** | `repo://libs/langchain_v1/tests/integration_tests/` | Integration tests for agent execution, tool binding, and provider fallback |
| **Tests (Partner Unit)** | `repo://libs/partners/*/tests/unit_tests/` | Provider-specific unit tests for chat models and integrations |
| **Tests (Partner Integration)** | `repo://libs/partners/*/tests/integration_tests/` | Provider-specific integration tests using live APIs (with cassettes for replay) |
| **Standard Tests** | `repo://libs/standard-tests/langchain_tests/` | Shared test suites and contracts for component conformance across the ecosystem |
| **Configuration (Core)** | `repo://libs/core/pyproject.toml` | langchain-core package metadata, dependencies (langsmith, httpx, tenacity, pydantic), and build config |
| **Configuration (LangChain v1)** | `repo://libs/langchain_v1/pyproject.toml` | langchain v1 package metadata, core dependencies, and optional provider groups |
| **Configuration (Classic)** | `repo://libs/langchain/pyproject.toml` | langchain-classic package metadata (legacy chains, memory, retrieval) |
| **Configuration (Text Splitters)** | `repo://libs/text-splitters/pyproject.toml` | langchain-text-splitters package metadata and dependencies |
| **Configuration (Repo-Wide)** | `repo://.pre-commit-config.yaml` | Git hooks for formatting, linting, and validation across all packages |
| **Build System (Libs)** | `repo://libs/Makefile` | Multi-package lockfile coordination, build targets, and cross-package tasks |

## Key Directory Structure

```
/libs/
├── core/                           # langchain-core: Base abstractions (v1.6.12)
│   ├── langchain_core/
│   │   ├── language_models/        # BaseChatModel and language model abstractions
│   │   │   ├── chat_models.py      # Core chat model interface
│   │   │   ├── base.py             # BaseLanguageModel and registry
│   │   │   └── model_profile.py    # Profile metadata integration
│   │   ├── messages/               # Message types and content blocks
│   │   │   ├── base.py             # BaseMessage and AIMessage
│   │   │   ├── content.py          # Content block types (text, images, tool use)
│   │   │   ├── ai.py               # AIMessage, AIMessageChunk
│   │   │   ├── tool.py             # ToolMessage, ToolCall handling
│   │   │   └── utils.py            # Message utilities and converters
│   │   ├── runnables/              # Runnable protocol and composition
│   │   │   ├── base.py             # Runnable[Input, Output] protocol
│   │   │   ├── config.py           # RunnableConfig and context management
│   │   │   ├── router.py           # Routing and branching logic
│   │   │   └── utils.py            # Composition utilities
│   │   ├── tools/                  # BaseTool and tool conversion
│   │   │   ├── base.py             # BaseTool abstract base
│   │   │   ├── convert.py          # Function/Pydantic to tool conversion
│   │   │   ├── structured.py       # StructuredTool with schema
│   │   │   └── render.py           # Tool rendering for models
│   │   ├── prompts/                # Prompt templates and composition
│   │   │   ├── base.py             # BasePrompt and PromptTemplate
│   │   │   ├── chat.py             # ChatPromptTemplate, message composition
│   │   │   ├── few_shot.py         # Few-shot example selection
│   │   │   └── image.py            # Image prompt support
│   │   ├── output_parsers/         # Output parsing and validation
│   │   │   ├── base.py             # BaseOutputParser interface
│   │   │   ├── pydantic.py         # Pydantic model parsing
│   │   │   ├── openai_tools.py     # OpenAI tool/function call parsing
│   │   │   ├── json.py             # JSON output parsing
│   │   │   └── xml.py              # XML structured output parsing
│   │   ├── callbacks/              # Callback manager and handlers
│   │   │   ├── manager.py          # CallbackManager, async variants, event coordination
│   │   │   ├── base.py             # BaseCallbackHandler interface
│   │   │   ├── streaming_stdout.py # Streaming output callback
│   │   │   └── file.py             # File-based logging
│   │   ├── language_models/        # Additional model utilities
│   │   ├── caches.py               # Cache interface for LLM results
│   │   ├── rate_limiters.py        # Rate limiter base and in-memory impl
│   │   ├── retrievers.py           # Retriever abstraction
│   │   ├── vectorstores/           # Vector store interface and in-memory impl
│   │   ├── documents/              # Document type and utilities
│   │   ├── embeddings/             # Embedding interface and fake impl
│   │   ├── document_loaders/       # Document loader abstractions
│   │   ├── example_selectors/      # Few-shot example selector strategies
│   │   └── ... (other utilities)
│   ├── tests/
│   │   ├── unit_tests/
│   │   ├── integration_tests/
│   │   └── benchmarks/
│   ├── Makefile
│   └── pyproject.toml
│
├── langchain_v1/                   # langchain: Orchestration and agents (v1.4.6)
│   ├── langchain/
│   │   ├── agents/
│   │   │   ├── factory.py          # Agent factory: graph construction, node orchestration
│   │   │   ├── middleware/         # Pluggable middleware system (20+ impls)
│   │   │   │   ├── types.py        # AgentMiddleware, AgentState, callbacks
│   │   │   │   ├── _retry.py       # Retry coordination
│   │   │   │   ├── _execution.py   # Execution lifecycle
│   │   │   │   ├── model_retry.py  # Model invocation retry
│   │   │   │   ├── tool_retry.py   # Tool call retry
│   │   │   │   ├── tool_call_limit.py # Tool invocation limits
│   │   │   │   ├── model_fallback.py # Model fallback chains
│   │   │   │   ├── human_in_the_loop.py # Human approval loops
│   │   │   │   ├── pii.py          # PII detection and redaction
│   │   │   │   ├── shell_tool.py   # Shell command execution
│   │   │   │   ├── file_search.py  # File system search and indexing
│   │   │   │   ├── tool_selection.py # LLM-based tool selection
│   │   │   │   ├── context_editing.py # Message context manipulation
│   │   │   │   ├── summarization.py # Conversation summarization
│   │   │   │   └── ... (other middleware)
│   │   │   ├── structured_output.py # Response schema marshaling
│   │   │   └── _subagent_transformer.py # Sub-agent graph composition
│   │   ├── chat_models/
│   │   │   └── base.py             # init_chat_model factory and registry
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
├── langchain/                      # langchain-classic: Legacy code (v1.0.8)
│   ├── langchain_classic/
│   │   ├── chains/                 # Legacy chain abstractions
│   │   ├── memory/                 # Conversation memory implementations
│   │   ├── llms/                   # Legacy LLM abstractions
│   │   ├── retrievers/             # Legacy retrieval abstractions
│   │   ├── document_loaders/       # Legacy document loading
│   │   └── ... (other legacy modules)
│   ├── tests/
│   ├── Makefile
│   └── pyproject.toml
│
├── partners/                       # Provider-specific integrations
│   ├── openai/                     # ChatOpenAI, embeddings, decisions API
│   │   ├── langchain_openai/
│   │   │   ├── chat_models/base.py # ChatOpenAI implementation
│   │   │   ├── decisions/          # Structured outputs integration
│   │   │   ├── embeddings/         # OpenAI embedding models
│   │   │   ├── llms/               # Legacy LLM compatibility
│   │   │   ├── output_parsers/     # Tool/function call parsing
│   │   │   ├── middleware/         # OpenAI-specific middleware
│   │   │   └── tools/              # OpenAI-specific tool integrations
│   │   ├── tests/
│   │   ├── Makefile
│   │   └── pyproject.toml
│   │
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
│   ├── together/                   # Together AI models
│   ├── google-vertexai/            # Google Vertex AI
│   ├── google-genai/               # Google Generative AI
│   ├── azure-ai/                   # Azure OpenAI and AI services
│   ├── aws/                        # AWS Bedrock and SageMaker
│   ├── chroma/                     # Chroma vector store
│   ├── qdrant/                     # Qdrant vector store
│   ├── exa/                        # Exa search
│   ├── nomic/                      # Nomic embeddings
│   ├── typesafe/                   # Typesafe schema generation
│   ├── cohere/                     # Cohere models
│   └── Makefile
│
├── text-splitters/                 # Text splitting utilities (v1.x)
│   ├── langchain_text_splitters/
│   │   ├── base.py                 # TextSplitter base class
│   │   ├── character.py            # Character-based splitting
│   │   ├── token_based.py          # Token count aware splitting
│   │   ├── semantic.py             # Semantic (embedding-based) splitting
│   │   └── language_specific.py    # Language-specific algorithms
│   ├── tests/
│   ├── Makefile
│   └── pyproject.toml
│
├── model-profiles/                 # LLM behavior and capability profiles (v0.0.6)
│   ├── langchain_model_profiles/
│   │   ├── cli.py                  # CLI for updating profiles
│   │   └── _summary.py             # Profile metadata aggregation
│   ├── tests/
│   ├── Makefile
│   └── pyproject.toml
│
├── standard-tests/                 # Cross-ecosystem test contracts
│   ├── langchain_tests/
│   │   ├── base.py                 # BaseStandardTests class
│   │   ├── unit_tests/             # Standard unit test suites
│   │   ├── integration_tests/      # Standard integration test suites
│   │   ├── conftest.py             # Pytest configuration
│   │   └── utils/                  # Test utilities
│   ├── tests/
│   ├── Makefile
│   └── pyproject.toml
│
├── Makefile                        # Multi-package build coordination
└── README.md
```

## Architecture Layers

### Layer 1: langchain-core (Base Abstractions)
Core interfaces and implementations used by all other packages:
- **Message protocol**: AIMessage, HumanMessage, SystemMessage, ToolMessage with rich content blocks
- **Runnable protocol**: Composable units with `invoke()`, `stream()`, `batch()` and async variants
- **LLM interface**: BaseChatModel for unified provider integration
- **Tool system**: BaseTool with schema generation and conversion utilities
- **Prompt templates**: ChatPromptTemplate and PromptTemplate for composition
- **Output parsing**: Structured output extraction (JSON, XML, Pydantic, OpenAI tools)
- **Callbacks**: Event instrumentation and LangSmith integration
- **Retrieval**: Document loaders, vectorstores, retrievers
- **Utilities**: Embeddings, caches, rate limiters, example selectors

### Layer 2: langchain (Orchestration & Agents)
High-level abstractions for agentic loops and orchestration:
- **Agent Factory** (`factory.py`): Constructs LangGraph state machines with 7 node types (entry, loop entry, model, after-model, tools, exit) orchestrated by middleware
- **Middleware System**: Pluggable hooks for model boundaries, tool boundaries, and lifecycle events (before_agent, before_model, after_model, after_tool, after_agent)
- **Model Initialization**: `init_chat_model()` factory for provider:model discovery and dynamic loading
- **Structured Output**: Schema marshaling for typed agent responses (Pydantic, dataclass, TypedDict, JSON schema)
- **MCP Integration**: Model Context Protocol adapter for tool discovery

### Layer 3: langchain-classic (Legacy Code)
Backward-compatible implementations for v0.x users:
- **Chains**: LLMChain, SequentialChain, and other legacy composition patterns
- **Memory**: BufferMemory, EntityMemory, summary-based memory implementations
- **Retrievers**: Legacy retriever abstractions
- **Loaders**: Legacy document loader patterns

### Layer 4: Partners (Provider Integrations)
Provider-specific implementations as independent packages:
- Each partner follows the **partner pattern**: extends core abstractions, released independently, dependencies pinned to core versions
- **ChatOpenAI, ChatAnthropic, ChatOllama, etc.**: Each implements BaseChatModel
- **Decisions API**: OpenAI-specific structured outputs integration
- **Vector Stores**: Chroma, Qdrant implementations of BaseVectorStore
- **Embeddings**: Provider-specific embedding models

## Common Workflows

### Finding Agent-Related Code
- **Agent construction**: `repo://libs/langchain_v1/langchain/agents/factory.py` (graph nodes, middleware orchestration)
- **Middleware hooks**: `repo://libs/langchain_v1/langchain/agents/middleware/types.py` (type definitions); individual middleware in `repo://libs/langchain_v1/langchain/agents/middleware/` subdirectory (20+ implementations)
- **Structured output**: `repo://libs/langchain_v1/langchain/agents/structured_output.py` (schema marshaling, response validation)
- **Tests**: `repo://libs/langchain_v1/tests/unit_tests/` and `repo://libs/langchain_v1/tests/integration_tests/`

### Finding LLM Integration Code
- **Provider implementations**: `repo://libs/partners/<provider>/` (e.g., `repo://libs/partners/openai/`)
  - Chat models: `langchain_*/chat_models/base.py`
  - Embeddings: `langchain_*/embeddings/`
  - Provider-specific middleware: `langchain_*/middleware/`
- **Model discovery**: `repo://libs/langchain_v1/langchain/chat_models/base.py` (`init_chat_model()` factory, registry)
- **Base interface**: `repo://libs/core/langchain_core/language_models/chat_models.py` (BaseChatModel protocol)
- **Model profiles**: `repo://libs/model-profiles/langchain_model_profiles/` (metadata registry and CLI)

### Finding Core Abstractions
- **Runnable protocol**: `repo://libs/core/langchain_core/runnables/base.py` (composition, streaming, batching)
- **Messages**: `repo://libs/core/langchain_core/messages/` (types, content blocks, utilities)
- **Tools**: `repo://libs/core/langchain_core/tools/base.py` (tool definition, schema, rendering)
- **Prompts**: `repo://libs/core/langchain_core/prompts/` (templates, composition, few-shot)
- **Callbacks**: `repo://libs/core/langchain_core/callbacks/` (event system, LangSmith integration)
- **Output parsing**: `repo://libs/core/langchain_core/output_parsers/` (structured extraction, validation)
- **Vector stores**: `repo://libs/core/langchain_core/vectorstores/base.py` (interface and in-memory impl)
- **Retrievers**: `repo://libs/core/langchain_core/retrievers.py` (document retrieval abstraction)

### Finding Tests
- **Core abstractions**: `repo://libs/core/tests/` (unit and integration tests for runnables, messages, tools, etc.)
- **Agent and orchestration**: `repo://libs/langchain_v1/tests/` (agent factory, middleware, initialization tests)
- **Provider-specific**: `repo://libs/partners/<provider>/tests/` (provider-specific unit and integration tests)
- **Shared test contracts**: `repo://libs/standard-tests/` (conformance tests for components across ecosystem)
- **Integration test data**: `repo://libs/langchain_v1/tests/cassettes/` (VCR cassettes for replaying HTTP interactions)

### Configuration and Build
- **Lint and format**: `.pre-commit-config.yaml` at repo root (ruff, black, mypy config)
- **Core dependencies**: `repo://libs/core/pyproject.toml` (langsmith, httpx, tenacity, pydantic)
- **LangChain dependencies**: `repo://libs/langchain_v1/pyproject.toml` (core, langgraph, optional provider groups)
- **Multi-package tasks**: `repo://libs/Makefile` (lock, check-lock targets for all packages)

### Adding a New Provider Integration
1. Create `repo://libs/partners/<provider>/` directory
2. Implement `langchain_<provider>/chat_models/base.py` extending `BaseChatModel`
3. Follow **partner pattern**: independent dependencies, core pinning, standard test suite
4. Add to `repo://libs/langchain_v1/langchain/chat_models/base.py` registry in `_BUILTIN_PROVIDERS`
5. Organize tests in `tests/unit_tests/` and `tests/integration_tests/`

### Adding a Middleware Implementation
1. Create new file in `repo://libs/langchain_v1/langchain/agents/middleware/`
2. Extend `AgentMiddleware` from `repo://libs/langchain_v1/langchain/agents/middleware/types.py`
3. Implement hook methods: `before_agent()`, `before_model()`, `after_model()`, `after_tool_call()`, `after_agent()`
4. Register in middleware composition chain via `factory.py` builder

## Build and Development Commands

All package directories (`libs/core/`, `libs/langchain_v1/`, `libs/partners/<provider>/`, etc.) include a local `Makefile` with standard targets:

```bash
# Format code (ruff)
make -C <package> format

# Lint code (ruff)
make -C libs/core lint

# Type check (mypy)
make -C libs/core type

# Run all tests
make -C libs/langchain_v1 test

# Run specific test file
make -C libs/core TEST_FILE=tests/unit_tests/test_runnables.py test

# Lock dependencies
make -C libs/core lock

# Check lockfile consistency
make -C libs/core check-lock

# Extended tests (marked with @pytest.mark.extended)
make -C libs/core extended_tests
```

The root `repo://libs/Makefile` coordinates multi-package operations:

```bash
# Lock all packages at once
make -C libs lock

# Check all lockfiles
make -C libs check-lock
```

## Key Implementation Details

### Agent Factory & Middleware Orchestration
The agent construction pipeline in `repo://libs/langchain_v1/langchain/agents/factory.py` builds a LangGraph state machine with these node types:
- **Entry**: Runs `before_agent` middleware once at start
- **Loop Entry**: Begins each model iteration, runs `before_model` hooks
- **Model**: Invokes language model with accumulated message history and tool bindings
- **After Model**: Runs `after_model` hooks for response processing, tool call extraction
- **Tools**: Executes tool calls in parallel when present
- **Exit**: Runs `after_agent` hooks once at completion

Middleware hooks are composable and execute in order:
- `before_agent(state, config)` → single entry point
- `before_model(state, config)` → per-iteration, before model call
- `after_model(response, state, config)` → after model call, can transform response
- `after_tool_call(tool_call, tool_result, state, config)` → per tool execution
- `after_agent(state, config)` → single exit point

Each middleware can inject commands to jump to different nodes, modify state, or abort execution.

### Middleware Implementations (20+ in `/agents/middleware/`)
- **Model-level**: `ModelRetryMiddleware`, `ModelCallLimitMiddleware`, `ModelFallbackMiddleware`
- **Tool-level**: `ToolRetryMiddleware`, `ToolCallLimitMiddleware`, `ToolErrorMiddleware`, `ToolEmulator`
- **Context**: `HumanInTheLoopMiddleware`, `ContextEditingMiddleware`, `FilesystemFileSearchMiddleware`
- **Safety/Redaction**: `PIIMiddleware`, `RedactionMiddleware`, `ShellToolMiddleware`
- **Execution Control**: `ProviderToolSearchMiddleware`
- **Optimization**: `SummarizationMiddleware`, `TodoListMiddleware`
- **Selection**: `LLMToolSelectorMiddleware`
- **Lifecycle**: `TracePolicy` configuration and execution tracing

Each middleware is independently configurable and composable into agent construction chains.

### Runnable Composition Protocol
The Runnable protocol in `repo://libs/core/langchain_core/runnables/base.py` enables declarative chaining via operators:
- **Piping** (`|`): Sequential composition, threads output to next input
- **Parallel** (`+`): Parallel execution of branches, collects results
- **Branching** (`.pipe()` with routing): Conditional execution paths based on state
- **Fallback** (`.with_fallback()`): Error recovery with alternative runnables
- **Retry** (`.with_retry()`): Automatic retry with exponential backoff

All compositions automatically support:
- `invoke(input, config=None)` → sync execution
- `ainvoke(input, config=None)` → async execution
- `batch(inputs, config=None, ...)` → parallel batch processing
- `stream(input, config=None)` → streaming output chunks
- `astream()`, `batch_as_completed()`, and other variants

### Message Protocol & Content Blocks
The message abstraction in `repo://libs/core/langchain_core/messages/` defines:
- **Message types**: `AIMessage`, `HumanMessage`, `SystemMessage`, `ToolMessage`, `FunctionMessage`
- **Content representation**: Messages store `content` as either:
  - `str` for simple text
  - `list[ContentBlock]` for mixed content (text, images, tool use, etc.)
- **Content blocks**: `TextBlock`, `ImageBlock`, `ToolUseBlock`, `ToolResultBlock`, custom blocks
- **Message utilities**: 
  - Serialization/deserialization via `dumps()`, `loads()`
  - Message merging and deduplication
  - Role mapping for different model APIs
  - Model-specific translation (e.g., OpenAI format conversion)

### Tool Abstraction & Binding
The `BaseTool` in `repo://libs/core/langchain_core/tools/base.py` provides:
- **Tool protocol**: 
  - Sync/async invoke methods
  - Schema generation from docstrings and Pydantic models
  - Argument validation and type coercion
- **Tool conversion** (via `convert.py`):
  - `tool()` decorator for functions → StructuredTool
  - Pydantic model → Tool with JSON schema
  - Custom tool builders
- **Tool rendering** (via `render.py`):
  - Format tools for model context
  - Description-only for prompt inclusion
  - Structured schemas for function calling

Tool binding in chat models:
```python
model.bind_tools([tool1, tool2])  # Binds tools via provider API
```

### Output Parser Hierarchy
- **BaseOutputParser**: Abstract base for all parsers
- **PydanticOutputParser**: Validates against Pydantic model
- **JsonOutputParser**: Parses JSON text into dict
- **XmlOutputParser**: Parses XML structured output
- **OpenAI tool/function parsers**: Handle tool call and function call formats
- **Custom parsers**: User-defined parsing logic

## Dependency Flow

```
User Applications
  ├─→ langchain (v1.4.6)
  │    ├─→ langchain-core (v1.6.12)
  │    └─→ LangGraph (state machines, runtime)
  │
  ├─→ langchain-core (direct use)
  │    └─→ LangSmith (tracing), httpx (HTTP), tenacity (retry), pydantic
  │
  ├─→ langchain-classic (v1.0.8) [backward compat]
  │    └─→ langchain-core, SQLAlchemy, LangSmith
  │
  └─→ Partner Packages (langchain-openai, langchain-anthropic, etc.)
       ├─→ langchain-core (pinned versions)
       └─→ Provider SDKs (openai, anthropic, etc.)
```

**Versioning Model**:
- Core is released independently with strict semantic versioning
- LangChain and partners pin core versions with lower bounds
- Partner packages are released independently per provider
- Standard-tests library contains shared conformance contracts

## Extension Points

### Adding Custom Chat Model
1. Extend `BaseChatModel` from `langchain_core.language_models.chat_models`
2. Implement: `_generate()`, `_stream()`, `_invoke()` methods
3. Register in `init_chat_model()` via `_BUILTIN_PROVIDERS` dict
4. Inherit callback support, streaming, token counting, structured output from base

### Adding Custom Retriever
1. Extend `BaseRetriever` from `langchain_core.retrievers`
2. Implement: `_get_relevant_documents()`, `_aget_relevant_documents()`
3. Automatic integration with callback system and streaming

### Adding Custom Output Parser
1. Extend `BaseOutputParser` from `langchain_core.output_parsers`
2. Implement: `parse()` method (and `aparse()` for async)
3. Optionally implement `get_format_instructions()` for prompt injection

### Adding Custom Tool
1. Extend `BaseTool` or use `@tool` decorator
2. Implement `_run()` (and `_arun()` for async)
3. Schema automatically generated from docstrings and Pydantic annotations

### Adding Custom Runnable
1. Extend `Runnable[Input, Output]` or `RunnableSerializable`
2. Implement: `invoke()`, `stream()`, `batch()` (async variants optional)
3. Automatically compose with operators: `|`, `+`, `.with_fallback()`, etc.
