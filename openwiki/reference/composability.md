---
type: Redirect
title: Composability (Deprecated Reference)
description: This page has been moved. For the authoritative documentation on agent composability, middleware composition, and subagent delegation patterns, see the Agent Composability documentation.
tags: [composability, redirect, deprecated, agents, middleware, subagents]
verified:
  - by: openwiki/0.5.0
    at: 2026-10-25T08:23:03.589Z
sources:
  - id: openwiki-source-2bb5c49861e72a9099b51970
    resource: repo://libs/langchain_v1/langchain/agents/_subagent_transformer.py
generated: { by: "openwiki/0.5.0", at: "2026-10-25T08:23:03.589Z" }
---

## Deprecated Page

This page has been superseded. For the authoritative documentation on agent composability, middleware composition, and subagent delegation, please see:

**→ [Agent Composability: Building Complex Agents](/openwiki/composability.md)**

The primary page contains comprehensive coverage of:

- **Middleware Composition** – Layering request/response handlers, middleware stack architecture, and lifecycle hooks
- **State Flow Through Middleware** – AgentState structure, runtime context, and state transformations
- **Subagent Delegation** – Tool-based agent dispatch, context propagation, and nested execution
- **Tool Selection Strategies** – Filtering tools, routing decisions, and dynamic tool injection
- **Parallel and Sequential Tool Execution** – LangGraph Send commands, batch operations, and flow control
- **Context Management** – Propagating context across agent boundaries and maintaining observability
- **Error Handling and Fallbacks** – Graceful degradation and retry patterns in composite agents
- **Observability and Tracing** – Monitoring hierarchical agent workflows and middleware interactions
