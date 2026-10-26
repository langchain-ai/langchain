---
type: "Redirect"
title: "Chat Models (Old Reference)"
description: "Deprecated reference page. Chat Models documentation has been consolidated into /openwiki/concepts/chat-models.md."
tags: ["deprecated", "redirect", "chat-models"]
verified:
  - by: openwiki/0.5.0
    at: 2026-10-25T08:23:03.589Z
sources:
  - id: openwiki-source-c52037e7b642f7ac5a7642a8
    resource: repo://libs/core/langchain_core/language_models/chat_models.py
generated: { by: "openwiki/0.5.0", at: "2026-10-25T08:23:03.589Z" }
---

# Chat Models (Old Reference)

This is a deprecated reference page that has been superseded by the canonical documentation. All current information about chat models, their architecture, integration points, and usage has been consolidated into a single authoritative source.

## See Instead

**→ [Chat Models: LLM Integration Points](/openwiki/concepts/chat-models.md)**

The canonical Chat Models concept page contains comprehensive documentation on LangChain's chat model architecture, including:

- Core `BaseChatModel` interface and implementation contract
- Message conversion, normalization, and provider-specific handling
- Streaming architecture (V1 callback-based and V2 protocol events)
- Tool calling integration: binding, schema conversion, and structured output patterns
- `AIMessage` structure and response metadata
- Caching, rate limiting, and generation lifecycle
- Batch generation with `generate()` and `agenerate()`
- Universal factory function `init_chat_model()` for provider-agnostic model instantiation

For all questions about chat models in LangChain, refer to the consolidated concept page instead of this deprecated reference.
