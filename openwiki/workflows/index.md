# Files

- [Agent Execution: Building Agentic Workflows](agent-execution.md) - End-to-end agent execution patterns using the Agent Factory and middleware composition. Covers agent definition, tool binding, middleware ordering, and common control flow patterns.
- [Composability: Building Modular Systems](composability.md) - Composition patterns for building complex applications from simple, reusable components. Covers Runnable piping, component interfaces, and composability boundaries.
- [Middleware: Extensible Request Processing](middleware.md) - Middleware system for composing agent behavior. Covers middleware interface, execution order, state management, and patterns for routing, logging, and tool control.
- [Model Initialization: Universal Model Loading](model-initialization.md) - The init_chat_model factory function and provider routing system. Covers model identifier syntax, provider discovery, fallback logic, and model profile integration for loading any supported LLM.
- [Structured Output: Type-Safe Results from Language Models](structured-output.md) - Patterns for obtaining structured, validated output from language models using Pydantic schemas, JSON schemas, and output parsers. Bind schemas to chat models with with_structured_output() or parse raw model responses using BaseOutputParser implementations.
