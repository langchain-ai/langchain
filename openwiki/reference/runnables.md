---
type: "Redirect"
title: "Runnables (Old Reference)"
description: "Deprecated reference page. Runnables documentation has been consolidated into /openwiki/concepts/runnables.md."
tags: ["deprecated", "redirect", "runnable", "lcel"]
verified:
  - by: openwiki/0.5.0
    at: 2026-10-25T08:23:03.589Z
sources:
  - id: openwiki-source-a1981e868973f6fd7f71e12e
    resource: repo://libs/core/langchain_core/runnables/base.py
generated: { by: "openwiki/0.5.0", at: "2026-10-25T08:23:03.589Z" }
---

# Runnables (Old Reference)

This is a deprecated reference page that has been superseded by the canonical documentation. All current information about the Runnable interface, execution patterns, and LCEL composition has been consolidated into a single authoritative source.

## See Instead

**→ [Runnables: Core Execution Abstraction](/openwiki/concepts/runnables.md)**

The canonical Runnables concept page contains comprehensive documentation on LangChain's foundational execution abstraction, including:

- Runnable interface design: the unified execution protocol for all LangChain components
- Core execution methods: synchronous and asynchronous invoke, batch processing, and streaming
- Default implementations and optimization points for custom subclasses
- Schema introspection: input/output types and configuration schemas
- LangChain Expression Language (LCEL) composition: declarative chaining with operators and methods
- Sequential piping (`|`), parallel composition (dict literals), branching, fallback handling, and retry logic
- Configuration threading: carrying tags, metadata, callbacks, and configurable parameters through chains
- Streaming architecture: incremental output generation and end-to-end chain streaming
- Transform method and advanced streaming patterns for progressive data flow

For all questions about Runnables, the execution model, and LCEL composition in LangChain, refer to the consolidated concept page instead of this deprecated reference.
