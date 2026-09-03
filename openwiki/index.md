---
okf_version: "0.2"
---

# Files

- [Agent Execution Flow and Loop Control](agent-execution.md) - Traces the runtime lifecycle of an agent from user input through model invocation, tool dispatch, and loop termination conditions, with detailed state management and middleware integration points.
- [Create a basic agent](agent-factory.md)
- [LangChain System Architecture](architecture.md) - High-level decomposition of the LangChain framework into three layers: langchain-core (abstractions), langchain (orchestration and agents), and partners (provider integrations), showing dependencies, component responsibilities, and extension boundaries.
- [> Entering new SequentialChain chain...](callbacks.md)
- [Chat Model Interface and Lifecycle](chat-models.md) - Document BaseChatModel protocol, input/output handling, streaming, and integration points with callbacks and model profiling.
- [CI/CD Workflows: GitHub Actions and Release Process](ci-workflows.md)
- [Dict syntax creates a RunnableParallel](composability.md)
- [Development Commands and Local Setup](dev-commands.md) - Quick reference for uv, make, lint, test, and type-checking commands in the LangChain monorepo, including environment setup, pre-commit hooks, and testing workflows.
- [Integration Testing: Live API Tests and VCR Cassettes](integration-tests.md) - How to write integration tests that call real model APIs with VCR cassette recording for CI compatibility, including environment setup, cassette management, and parameterization patterns.
- [Bearer token](mcp-integration.md)
- [Message Types and Content Representation](messages.md) - Document the message abstraction, standardized content blocks for multimodal LLM I/O, message hierarchy, and provider-specific block translators.
- [Middleware](middleware.md)
- [Chat Model Initialization with init_chat_model](model-initialization.md) - Factory function for instantiating chat models from provider strings with unified configuration and runtime model switching.
- [Use async methods (ainvoke, astream)](openai-provider.md)
- [Adding a New Chat Model Provider](partner-pattern.md) - Step-by-step guide to integrate a new LLM provider into LangChain's monorepo, including package structure, ChatModel implementation, streaming, function calling, and standard tests.
- [Prompt Templates and Few-Shot Learning](prompts.md) - Prompt templates define message sequences and variable substitution patterns for chat models. Few-shot learning selects examples dynamically to teach models by example.
- [LangChain Repository Quick Start](quickstart.md) - Entry point for engineers: orient to the monorepo structure, run first tests, understand what to edit for common tasks, and route to major development areas.
- [Runnable: Core Composition Layer](runnables.md) - Explain the Runnable protocol and how it enables composable chaining of LLM components through the LangChain Expression Language (LCEL).
- [Source Map: Repository File Organization](source-map.md) - Quick reference for locating code by topic, mapping LangChain concepts to their implementation paths across the monorepo including core abstractions, agents, middleware, partners, and configuration files.
- [Streaming: Token-by-Token Output](streaming.md) - How streaming works across LLM components and chains, token-by-token delivery via AIMessageChunk, callback integration, and memory/latency tradeoffs.
- [AutoStrategy (recommended)](structured-output.md)
- [Form 1: No arguments (name from function)](tools.md)
- [Unit Testing: Strategies and Patterns](unit-tests.md) - How to write unit tests for langchain-core and langchain components using pytest, fixtures, mocking, and standard test classes from langchain-tests.
