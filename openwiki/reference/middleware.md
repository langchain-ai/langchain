---
type: Redirect
title: "Middleware (Deprecated Reference)"
description: "Deprecated reference page. See /openwiki/middleware.md for comprehensive documentation on agent middleware, lifecycle hooks, and composition patterns."
tags: [agent-middleware, redirect, deprecated]
verified:
  - by: openwiki/0.5.0
    at: 2026-10-25T08:23:03.589Z
sources:
  - id: openwiki-source-71e882e1ac9757ea8e959a7c
    resource: repo://libs/langchain_v1/langchain/agents/factory.py
  - id: openwiki-source-03e8ca0eebe37feda8566793
    resource: repo://libs/langchain_v1/langchain/agents/middleware/types.py
generated: { by: "openwiki/0.5.0", at: "2026-10-25T08:23:03.589Z" }
---

## Deprecated Reference

This reference page is deprecated and superseded by the canonical documentation at **[/openwiki/middleware.md](/openwiki/middleware.md)**.

Agent middleware in LangChain agents provides a composable, layered approach to intercepting and modifying agent behavior at well-defined execution points without changing core agent logic. Middleware registers handler pairs for lifecycle hooks spanning initialization (`before_agent`), per-iteration model invocation (`before_model`, `wrap_model_call`, `after_model`), tool execution (`wrap_tool_call`), and completion (`after_agent`), composing by wrapping handlers in registration order to enable cross-cutting concerns such as automatic retries, error handling, human-in-the-loop approval, and request/response transformation.

For complete documentation covering middleware composition semantics, handler interception patterns, state transformation, error propagation, and implementation patterns for common middleware use cases, please refer to the primary page linked above.
