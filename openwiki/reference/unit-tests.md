---
type: Redirect
title: Unit Testing and Mocking (Deprecated Reference)
description: This page has been superseded. See the primary Unit Testing and Mocking documentation for current content.
tags: [unit-tests, redirect, deprecated]
verified:
  - by: openwiki/0.5.0
    at: 2026-10-25T08:23:03.589Z
sources:
  - id: openwiki-source-727aef6a92fb635fdbb41cd6
    resource: repo://libs/core/tests/unit_tests/conftest.py
  - id: openwiki-source-3eb9100e02f9d70098d1b30d
    resource: repo://libs/standard-tests/langchain_tests/unit_tests/chat_models.py
generated: { by: "openwiki/0.5.0", at: "2026-10-25T08:23:03.589Z" }
---

## Deprecated Page

This page has been moved. For the authoritative documentation on unit testing, fixtures, mocking, and test patterns, please see:

**→ [Unit Testing and Mocking](/openwiki/unit-tests.md)**

The primary page contains comprehensive coverage of:

- **Test Structure & Organization** – Directory layout, running tests, parallelization, and socket restrictions
- **Standard Test Classes** – ChatModelUnitTests, ToolUnitTests, and EmbeddingUnitTests for consistent testing across integrations
- **Fixtures and Conftest** – Shared test fixtures, database fixtures, model fixtures, and configuration management
- **Mocking Strategies** – Mocking chat models, callbacks, tools, message handling, and async patterns
- **Agent Testing** – State transitions, middleware behavior, tool execution, and message flow
- **Assertions and Matchers** – Message validation, content block assertions, and tool call verification
- **Async Testing** – Async fixtures, event loop management, and concurrent execution patterns
- **Callbacks and Observability** – Tracking test execution, async callback managers, and trace verification
- **Tool Testing** – Argument schemas, invocation patterns, and error handling
- **Integration Patterns** – Testing with embeddings, chat models, and tool combinations
