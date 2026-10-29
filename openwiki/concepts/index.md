# Files

- [Create an LLM run](callbacks.md)
- [Chat Models: Core Abstractions and Protocol](chat-models.md) - Chat models are the fundamental abstraction for conversational AI in LangChain. This page documents the base class contract, lifecycle phases, message I/O semantics, streaming protocol, and integration boundaries.
- [Messages: Content Representation](messages.md) - Message types, content blocks, and multimodal support. Covers message hierarchy, role semantics, content block structure, streaming concatenation semantics, and provider-specific translation.
- [Prompts: Templating and Composability](prompts.md) - Prompt templates define message sequences and variable substitution patterns for language models. Covers template syntax, PromptValue routing, runnable integration, and few-shot learning with dynamic example selection.
- [Runnables: Core Execution Abstraction](runnables.md) - Runnable is the foundational execution abstraction in LangChain. It defines a unified, composable interface for synchronous, asynchronous, batch, and streaming execution. Every LangChain component—prompts, models, parsers, retrievers, tools—is a Runnable, enabling declarative composition and uniform control flow.
- [Bare decorator - infers schema from function signature](tools.md)
