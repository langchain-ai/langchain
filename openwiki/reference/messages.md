---
type: "Redirect"
title: "Messages (Old Page Reference)"
description: "Deprecated: See /openwiki/concepts/messages.md instead. This page has been moved to the concepts section."
tags: ["deprecated", "redirect", "messages"]
verified:
  - by: openwiki/0.5.0
    at: 2026-10-25T08:23:03.589Z
sources:
  - id: openwiki-source-85004ebd4eba7aec1e145503
    resource: repo://libs/langchain/langchain_classic/schema/messages.py
generated: { by: "openwiki/0.5.0", at: "2026-10-25T08:23:03.589Z" }
---

# Messages (Old Page Reference)

> **This page is deprecated.** The authoritative documentation has moved to [`/openwiki/concepts/messages.md`](/openwiki/concepts/messages.md).

## About This Redirect

Messages are the fundamental data contract for LLM communication, implementing a type hierarchy from `BaseMessage` with variants like `HumanMessage`, `AIMessage`, `SystemMessage`, and `ToolMessage`. This reference page has been consolidated into the concepts documentation for better organization and maintainability.

Please refer to `/openwiki/concepts/messages.md` for complete information about:
- The message type hierarchy and base message contract
- Standard message types (HumanMessage, AIMessage, SystemMessage, ToolMessage, etc.)
- Tool-related message types and tool call semantics
- Streaming variants and chunk concatenation
- Content blocks and multimodal message handling
- Message serialization and provider-specific conversions
