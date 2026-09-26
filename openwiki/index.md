---
okf_version: "0.2"
---

# Files

- [Agent Execution Flow and Loop Control](agent-execution.md) - Traces the runtime lifecycle of an agent from user input through model invocation, tool dispatch, and loop termination conditions, with detailed state management and middleware integration points.
- [Agent Factory and create_agent](agent-factory.md) - The agent factory constructs state machines that orchestrate conversation flow between a language model, tool execution, and middleware layers. The create_agent function handles tool binding, structured output, state schema resolution, and graph compilation.
- [LangChain System Architecture](architecture.md) - High-level decomposition of the LangChain framework into three layers: langchain-core (abstractions), langchain (orchestration and agents), and partners (provider integrations), showing dependencies, component responsibilities, and extension boundaries.
- [Callback System and Handler Integration](callbacks.md) - Document the callback handler architecture, integration with runnables and chat models, and patterns for tracking execution events, streaming, and instrumentation.
- [Chat Model Interface and Lifecycle](chat-models.md) - Document BaseChatModel protocol, input/output handling, streaming, and integration points with callbacks and model profiling.
- [CI/CD Workflows: GitHub Actions and Release Process](ci-workflows.md)
- [Composability and LCEL Chains](composability.md) - How Runnable components compose through LCEL operators, creating reusable workflows with automatic async, batch, and streaming support.
- [Development Commands and Local Setup](dev-commands.md) - Quick reference for uv, make, lint, test, and type-checking commands in the LangChain monorepo, including environment setup, pre-commit hooks, and testing workflows.
- [Integration Testing: Live API Tests and VCR Cassettes](integration-tests.md) - How to write integration tests that call real model APIs with VCR cassette recording for CI compatibility, including environment setup, cassette management, and parameterization patterns.
- [MCP (Model Context Protocol) Integration](mcp-integration.md) - LangChain adapter for discovering and invoking MCP tools, with protocol negotiation, multiple transports, mid-call interrupts, and error handling for agent use.
- [Message Types and Content Representation](messages.md) - Document the message abstraction, standardized content blocks for multimodal LLM I/O, message hierarchy, and provider-specific block translators.
- [Agent Middleware: Composable Request/Response Processing](middleware.md) - Document the middleware system for agents, including lifecycle hooks, HITL approval, error handling, retry logic, and middleware composition patterns for intercepting and modifying agent behavior.
- [Chat Model Initialization with init_chat_model](model-initialization.md) - Factory function for instantiating chat models from provider strings with unified configuration and runtime model switching.
- [OpenAI Integration: ChatOpenAI and Azure Support](openai-provider.md) - ChatOpenAI integration for OpenAI's Chat Completions and Responses APIs, with support for tool calling, structured output, vision, streaming, and Azure deployment.
- [Adding a New Chat Model Provider](partner-pattern.md) - Step-by-step guide to integrate a new LLM provider into LangChain's monorepo, including package structure, ChatModel implementation, streaming, function calling, structured output, and standard tests. Covers message conversion, error handling, model profiles, and optional advanced API modes like Responses API.
- [Prompt Templates and Few-Shot Learning](prompts.md) - Prompt templates define message sequences and variable substitution patterns for chat models. Few-shot learning selects examples dynamically to teach models by example.
- [LangChain Repository Quick Start](quickstart.md) - Entry point for engineers: orient to the monorepo structure, run first tests, understand what to edit for common tasks, and route to major development areas.
- [Runnable: Core Composition Layer](runnables.md) - Explain the Runnable protocol and how it enables composable chaining of LLM components through the LangChain Expression Language (LCEL).
- [Source Map: Repository File Organization](source-map.md) - Quick reference for locating code by topic, mapping LangChain concepts to their implementation paths across the monorepo including core abstractions, agents, middleware, partners, and configuration files.
- [Streaming: Token-by-Token Output](streaming.md) - How streaming works across LLM components and chains, token-by-token delivery via AIMessageChunk, callback integration, and memory/latency tradeoffs.
- [Structured Output: Binding Schemas and Response Marshaling](structured-output.md) - Mechanisms for binding Pydantic schemas and JSON schemas to LLM responses via tool-based, provider-native, or automatically-detected strategies; includes validation, error handling, and response marshaling.
- [Tools and Tool Binding](tools.md) - LangChain's tool system enables agents and language models to execute structured actions through schema-aware components with automatic validation, error handling, and callback integration.
- [Unit Testing: Strategies and Patterns](unit-tests.md) - How to write unit tests for langchain-core and langchain components using pytest, fixtures, mocking, and standard test classes from langchain-tests.
