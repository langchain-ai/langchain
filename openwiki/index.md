---
okf_version: "0.2"
---

# Files

- [Agent Execution Flow and Control](agent-execution.md) - Detailed walkthrough of agent execution from user input through model invocation, tool dispatch, loop control, and termination conditions, including state management, middleware hooks, invocation patterns, and streaming modes.
- [Create a basic agent](agent-factory.md) - Document the Agent Factory as the foundational entry point for building LangChain agents, including create_agent function, state machine architecture, middleware composition, and common patterns.
- [LangChain System Architecture](architecture.md) - High-level decomposition of the LangChain framework into three layers: langchain-core (abstractions), langchain (orchestration and agents), and partners (provider integrations), showing dependencies, component responsibilities, and extension boundaries.
- [Callback System and Handler Integration](callbacks.md) - Document the callback handler architecture, integration with runnables and chat models, and patterns for tracking execution events, streaming, and instrumentation.
- [Chat Model Interface and Lifecycle](chat-models.md) - Document BaseChatModel protocol, input/output handling, streaming, and integration points with callbacks and model profiling.
- [CI/CD Workflows and Release Process](ci-workflows.md) - Overview of GitHub Actions workflows for testing, releasing packages, and updating documentation, including dependencies, matrix configurations, and release automation.
- [Building Composable Systems](composability.md) - Best practices for composable architecture: how to design separation of concerns, implement dependency injection, leverage type safety, and extend LangChain components through composition.
- [Development Commands and Local Setup](dev-commands.md) - Quick reference for uv, make, lint, test, and type-checking commands in the LangChain monorepo, including environment setup, pre-commit hooks, and testing workflows.
- [Integration Testing: Live API Tests and VCR Cassettes](integration-tests.md) - How to write integration tests that call real model APIs with VCR cassette recording for CI compatibility, including environment setup, cassette management, and parameterization patterns.
- [MCP (Model Context Protocol) Integration](mcp-integration.md) - The langchain.mcp module discovers and adapts MCP tools for LangChain agents, supporting multiple transports, elicitation-driven interrupts, tool error recovery, and multi-server prefixing.
- [Message Types and Content Representation](messages.md) - Document the message abstraction, standardized content blocks for multimodal LLM I/O, message hierarchy, and provider-specific block translators.
- [Agent Middleware: Composable Request/Response Processing](middleware.md) - Document the middleware system for agents, including lifecycle hooks, HITL approval, error handling, retry logic, and middleware composition patterns for intercepting and modifying agent behavior.
- [Chat Model Initialization with init_chat_model](model-initialization.md) - Factory function for instantiating chat models from provider strings with unified configuration and runtime model switching.
- [OpenAI Integration: ChatOpenAI and Azure Support](openai-provider.md) - ChatOpenAI integration for OpenAI's Chat Completions and Responses APIs, with support for tool calling, structured output, vision, streaming, and Azure deployment.
- [Adding a New Chat Model Provider](partner-pattern.md) - Step-by-step guide to integrate a new LLM provider into LangChain's monorepo, including package structure, ChatModel implementation, streaming, function calling, structured output, and standard tests. Covers message conversion, error handling, model profiles, and optional advanced API modes like Responses API.
- [Prompt Templates and Few-Shot Learning](prompts.md) - Prompt templates define message sequences and variable substitution patterns for chat models. Few-shot learning selects examples dynamically to teach models by example.
- [LangChain Repository Quick Start](quickstart.md) - Entry point for engineers: orient to the monorepo structure, run first tests, understand what to edit for common tasks, and route to major development areas.
- [Runnable: Core Composition Layer](runnables.md) - Explain the Runnable protocol and how it enables composable chaining of LLM components through the LangChain Expression Language (LCEL).
- [Source Map: Repository File Organization](source-map.md) - Quick reference for locating code by topic, mapping LangChain concepts to their implementation paths across the monorepo including core abstractions, agents, middleware, partners, and configuration files.
- [Streaming: Incremental Output and Real-Time Control](streaming.md) - Document streaming modes, token-by-token incremental delivery, asynchronous patterns, backpressure control, and client patterns for interactive agent feedback and responsive UIs.
- [Structured Output: Typed Agent Responses](structured-output.md) - Guide to structured output in agents—response schema definition, provider strategies, retry on validation failure, and type-safe response handling across models.
- [Tools: Defining, Converting, and Calling](tools.md) - Comprehensive guide to LangChain's tool system: converting functions to tools via decorators, input schema generation and validation, execution lifecycle, error handling, tool organization, and integration with agents.
- [Unit Testing: Strategies and Patterns](unit-tests.md) - How to write unit tests for langchain-core and langchain components using pytest, fixtures, mocking, and standard test classes from langchain-tests.
