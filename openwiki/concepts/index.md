# Files

- [Create an LLM run](callbacks.md)
- [Chat Models: LLM Integration Points](chat-models.md)
- [Messages: LLM Communication Protocol](messages.md) - Message types and structures—the data contract between application code and language models. Covers base message types, role semantics, tool-related message types, and provider-specific conversions.
- [Prompts: Templating and Composability](prompts.md) - Prompt templates define message sequences and variable substitution patterns for language models. Covers template syntax, PromptValue routing, runnable integration, and few-shot learning with dynamic example selection.
- [Runnables: Core Execution Abstraction](runnables.md) - Runnable is the foundational execution abstraction in LangChain. It defines a unified, composable interface for synchronous, asynchronous, batch, and streaming execution. Every LangChain component—prompts, models, parsers, retrievers, tools—is a Runnable, enabling declarative composition and uniform control flow.
- [Bare decorator - infers schema from function signature](tools.md)
