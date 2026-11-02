---
type: Redirect
title: Agent Execution (Deprecated Reference)
description: This page has been superseded. See the primary Agent Execution Flow and Loop Control documentation for current content.
tags: [agent-execution, redirect, deprecated]
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

## Deprecated Page

This page has been moved. For the authoritative documentation on agent execution, loop control, state management, and middleware integration, please see:

**→ [Agent Execution Flow and Loop Control](/openwiki/agent-execution.md)**

The primary page contains comprehensive coverage of:

- **Core Execution Pattern** – Initialization, model invocation, tool dispatch, result processing, and termination
- **Agent State** – State structure, message encoding, and reducer-based accumulation
- **Model Request and Response** – ModelRequest, ModelResponse, and ExtendedModelResponse objects
- **Middleware Integration** – Hook lifecycle, composition, and jump control
- **Loop Control and Termination** – Conditional edges and termination conditions
- **Tool Execution** – Tool call dispatch, execution, structured output tools
- **Graph Structure** – Nodes, entry/exit points, and conditional edges
- **Structured Output Processing** – Tool and provider strategies
- **Error Handling** – Model invocation, tool execution, and validation errors
- **Callbacks and Monitoring** – Observability and trace policies
