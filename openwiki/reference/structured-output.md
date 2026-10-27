---
type: Redirect
title: Structured Output (Deprecated Reference)
description: This page has been superseded. See the primary Structured Output and Response Formatting documentation for current content.
tags: [structured-output, redirect, deprecated]
verified:
  - by: openwiki/0.5.0
    at: 2026-10-25T08:23:03.589Z
sources:
  - id: openwiki-source-71e882e1ac9757ea8e959a7c
    resource: repo://libs/langchain_v1/langchain/agents/factory.py
  - id: openwiki-source-ec30ab6256dd50cc670919f6
    resource: repo://libs/langchain_v1/langchain/agents/structured_output.py
generated: { by: "openwiki/0.5.0", at: "2026-10-25T08:23:03.589Z" }
---

## Deprecated Page

This page has been moved. For the authoritative documentation on structured output, response format strategies, and schema validation, please see:

**→ [Structured Output and Response Formatting](/openwiki/structured-output.md)**

The primary page contains comprehensive coverage of:

- **Core Concept** – Response format specification and schema validation mechanisms
- **Response Format Strategies** – Tool-based, provider-native, and automatic strategy selection
- **ToolStrategy** – Tool-based structured output with Pydantic validation and error handling
- **ProviderStrategy** – Native structured output using provider APIs (OpenAI, Claude, etc.)
- **AutoStrategy** – Automatic detection and fallback between strategies
- **Parsing and Validation** – Schema conversion, JSON parsing, and validation workflows
- **Error Handling** – Retry behavior, exception handling, and error recovery
- **Middleware Integration** – Hook lifecycle and response processing
- **Configuration and Testing** – Schema registration, capability detection, and validation
