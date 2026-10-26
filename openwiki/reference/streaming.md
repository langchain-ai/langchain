---
type: Redirect
title: Streaming (Deprecated Reference)
description: This reference page is deprecated. See /openwiki/streaming.md for comprehensive documentation on token-by-token streaming, streaming architecture, and integration patterns.
tags: [streaming, token-streaming, redirect, deprecated]
verified:
  - by: openwiki/0.5.0
    at: 2026-10-25T08:23:03.589Z
sources:
  - id: openwiki-source-c52037e7b642f7ac5a7642a8
    resource: repo://libs/core/langchain_core/language_models/chat_models.py
generated: { by: "openwiki/0.5.0", at: "2026-10-25T08:23:03.589Z" }
---

## Deprecated Page

This page has been moved. For the authoritative documentation on streaming mechanics, token-by-token delivery, and streaming integration patterns, please see:

**→ [Streaming: Token-by-Token Output](/openwiki/streaming.md)**

The primary page contains comprehensive coverage of:

- **Overview** – How streaming works for incremental token delivery and real-time feedback
- **Synchronous Streaming** – The `stream()` entry point, control flow, chunk handling, and fallback behavior
- **Asynchronous Streaming** – The `astream()` variant, async/await patterns, and consumption strategies
- **Message Chunks** – Chunk types (AIMessageChunk, HumanMessageChunk, etc.) and mergeable composition
- **Callback Integration** – Event lifecycle, token-level callbacks, and error handling
- **Tool Call Streaming** – Incremental tool call parsing and finalization
- **Memory and Latency Tradeoffs** – Storage implications and performance considerations
- **Parser and Runnable Integration** – Streaming through chains and components
