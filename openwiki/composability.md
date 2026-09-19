---
type: "Concept"
title: "Building Composable Systems"
description: "Best practices for composable architecture: how to design separation of concerns, implement dependency injection, leverage type safety, and extend LangChain components through composition."
tags: [composition, runnable, lcel, type-safety, dependency-injection, middleware, extensibility, chaining]
verified:
  - by: openwiki/0.5.0
    at: 2026-09-19T08:23:50.449Z
sources:
  - id: openwiki-source-a1981e868973f6fd7f71e12e
    resource: repo://libs/core/langchain_core/runnables/base.py
  - id: openwiki-source-48e94bbe49ab4f33ba87e9cb
    resource: repo://libs/core/langchain_core/runnables/branch.py
  - id: openwiki-source-079792f059657900794e2955
    resource: repo://libs/core/langchain_core/runnables/config.py
  - id: openwiki-source-de6c904bd0171642bd50f6d9
    resource: repo://libs/core/langchain_core/runnables/router.py
  - id: openwiki-source-71e882e1ac9757ea8e959a7c
    resource: repo://libs/langchain_v1/langchain/agents/factory.py
generated: { by: "openwiki/0.5.0", at: "2026-09-19T08:23:50.449Z" }
---

## Overview

Composability is the core feature of LangChain systems: the ability to declaratively build complex workflows by combining small, reusable, independently testable components. The **Runnable protocol** at the core of LangChain provides the foundation; the **LangChain Expression Language (LCEL)** enables declarative composition using operators; and **middleware** provides a pluggable pattern for cross-cutting concerns.

This page covers the architectural patterns and practical techniques for building composable systems: how to structure components for reuse, how dependency injection works, how to leverage type safety with generics, and how to extend and customize behavior through middleware and composition.

## The Runnable Protocol as a Composition Foundation

Every composable component in LangChain implements the `Runnable` protocol, which defines a standardized interface for transformation:

```python
from langchain_core.runnables import Runnable, RunnableConfig

class Runnable(ABC, Generic[Input, Output]):
    """A unit of work that can be invoked, batched, streamed, transformed, and composed."""
    
    @abstractmethod
    def invoke(self, input: Input, config: RunnableConfig | None = None) -> Output:
        """Transform a single input synchronously."""
    
    async def ainvoke(self, input: Input, config: RunnableConfig | None = None) -> Output:
        """Async version; default delegates to invoke via executor."""
    
    def batch(self, inputs: list[Input], config: RunnableConfig | list[RunnableConfig] | None = None) -> list[Output]:
        """Process multiple inputs in parallel (default: thread pool)."""
    
    async def abatch(self, inputs: list[Input], config: RunnableConfig | Sequence[RunnableConfig] | None = None) -> list[Output]:
        """Async batch processing via asyncio.gather."""
    
    def stream(self, input: Input, config: RunnableConfig | None = None) -> Iterator[Output]:
        """Yield output chunks as produced (default: calls invoke once)."""
    
    async def astream(self, input: Input, config: RunnableConfig | None = None) -> AsyncIterator[Output]:
        """Async streaming."""
    
    @property
    def input_schema(self) -> TypeBaseModel:
        """Pydantic model describing input type and constraints."""
    
    @property
    def output_schema(self) -> TypeBaseModel:
        """Pydantic model describing output type."""
```

**Key principle**: Any component that implements `Runnable[Input, Output]` automatically inherits sync, async, batch, and streaming support. This uniformity enables composition without reimplementing execution modes.

## Sequential Composition: The Pipe Operator

The **pipe operator** (`|`) is the primary composition primitive. It chains Runnables sequentially, with each step's output becoming the next input:

```python
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Simple example
add_one = RunnableLambda(lambda x: x + 1)
mul_two = RunnableLambda(lambda x: x * 2)
sequence = add_one | mul_two
sequence.invoke(5)  # (5 + 1) * 2 = 12

# Real-world example: Prompt → Model → Parser
prompt = ChatPromptTemplate.from_template("Summarize: {text}")
model = ChatOpenAI()
parser = StrOutputParser()

chain = prompt | model | parser
output = chain.invoke({"text": "..."})
```

The `|` operator creates a **`RunnableSequence`**, which:

- **Flattens nested sequences** for efficiency: `(a | b) | (c | d)` becomes a single flat list `[a, b, c, d]`, not a tree.
- **Automatically preserves execution modes**: A sequence of Runnables that all support streaming will stream end-to-end; if any step blocks, the sequence blocks at that step.
- **Delegates batching intelligently**: Each step in a sequence batches its inputs independently, and outputs of one step become inputs to the next.
- **Supports sync and async**: The same chain works with `invoke`, `ainvoke`, `batch`, `abatch`, `stream`, and `astream` without modification.

## Parallel Composition: Branching and Fan-Out

Multiple Runnables can execute concurrently on the same input using **dict literals** (which create a `RunnableParallel`) or **explicit `RunnableParallel`**:

### Dict Literal Syntax

```python
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# Create independent branches from a shared input
chain = (
    RunnablePassthrough()  # Pass input through unchanged
    | {
        "summary": RunnableLambda(summarize_text),
        "entities": RunnableLambda(extract_entities),
        "sentiment": RunnableLambda(analyze_sentiment),
    }
)

result = chain.invoke("The quick brown fox...")
# {'summary': '...', 'entities': [...], 'sentiment': 'positive'}
```

### Explicit `RunnableParallel`

```python
from langchain_core.runnables import RunnableParallel

parallel = RunnableParallel(
    summary=summarize,
    entities=extract,
    sentiment=analyze,
)
result = parallel.invoke(text)
```

**Concurrency implementation**:

- **`RunnableParallel.invoke()`** uses `safetee` (sync) to create independent input copies for each branch, then executes them sequentially through the thread pool.
- **`RunnableParallel.ainvoke()`** uses `atee` (async) to create independent async iterators and runs branches concurrently with `asyncio.gather`.
- **Streaming**: When multiple branches stream, `RunnableParallel` yields chunks from whichever branch completes first, coordinated via `asyncio.wait(return_when=FIRST_COMPLETED)`.

## Batching for Parallel Input Processing

Batching processes multiple inputs through the same pipeline in parallel. It differs from parallel branching: branching runs different Runnables on one input; batching runs the same Runnable on many inputs.

```python
chain = prompt | model | parser

# Batch 10 texts through the same chain in parallel
results = chain.batch([text1, text2, ..., text10])  # Uses thread pool
await chain.abatch([text1, text2, ..., text10])  # Uses asyncio.gather
```

**Implementation**:

- **`batch()`** calls `invoke` for each input in parallel via a `ThreadPoolExecutor` (default) or a custom executor specified in config.
- **`abatch()`** uses `asyncio.gather` with concurrency control via the `max_concurrency` parameter in `RunnableConfig`.
- **`RunnableSequence.batch()`** calls `batch` on each step in order, feeding the batched outputs of one step as batched inputs to the next.
- Return exceptions as-is if `return_exceptions=True`; otherwise raise on first error.

## Conditional Routing: Branching Logic

Conditional routing selects different Runnables based on input predicates or key values.

### `RunnableBranch`: Predicate-Based Routing

```python
from langchain_core.runnables import RunnableBranch

router = RunnableBranch(
    (lambda x: "math" in x.lower(), math_chain),
    (lambda x: "code" in x.lower(), code_chain),
    (lambda x: "sql" in x.lower(), sql_chain),
    fallback_chain,  # Default if no condition matches
)

result = router.invoke("How do I calculate factorial?")  # → math_chain
```

Conditions are evaluated in order; the first truthy predicate selects its corresponding Runnable. If none match, the fallback branch runs.

### `RouterRunnable`: Key-Based Routing

```python
from langchain_core.runnables.router import RouterRunnable

router = RouterRunnable(
    runnables={
        "add": RunnableLambda(lambda x: x + 1),
        "multiply": RunnableLambda(lambda x: x * 2),
    }
)

result = router.invoke({"key": "multiply", "input": 5})  # 10
```

The input is a dict with `"key"` (selects which Runnable) and `"input"` (the data).

## Type Safety and Schema Inference

Composed chains automatically infer input and output types from their components, enabling validation without explicit annotations:

```python
from langchain_core.runnables import RunnableLambda

add_one = RunnableLambda(lambda x: x + 1)
mul_two = RunnableLambda(lambda x: x * 2)
chain = add_one | mul_two

# Inspect schemas
print(chain.input_schema)        # Pydantic model for input
print(chain.output_schema)       # Pydantic model for output
print(chain.input_schema.model_json_schema())  # JSON schema for docs/validation
```

### Generic Type Parameters

Runnables are generic over input and output types, enabling type checkers to catch errors:

```python
from langchain_core.runnables import Runnable

def my_chain() -> Runnable[dict[str, str], int]:
    """This chain accepts a string dict and returns an int."""
    return (
        RunnableLambda(lambda d: d["text"])
        | RunnableLambda(lambda s: len(s))
    )

chain = my_chain()
result = chain.invoke({"text": "hello"})  # Type: int
# chain.invoke(5)  # Type error: int is not dict[str, str]
```

### Schema Extension in Middleware and Complex Workflows

Components can declare custom state schemas, which are merged during graph construction:

```python
from langchain.agents.middleware.types import AgentMiddleware
from typing import TypedDict

class MyMiddleware(AgentMiddleware):
    state_schema: type[TypedDict] = MyStateExtension  # Merged into agent state
    
    def before_agent(self, state, runtime):
        # state now includes fields from MyStateExtension
        return {"custom_field": "value"}
```

## Dependency Injection and Configuration

The `RunnableConfig` system enables runtime configuration without modifying component code. Configuration flows through the execution context and is accessible to all components:

```python
from langchain_core.runnables import RunnableConfig, RunnableLambda

def my_func(x: int, config: RunnableConfig | None = None) -> int:
    """Access configuration at runtime."""
    if config:
        callbacks = config.get("callbacks", [])
        tags = config.get("tags", [])
        metadata = config.get("metadata", {})
    return x + 1

chain = RunnableLambda(my_func)

# Inject config at invocation time
result = chain.invoke(5, config={
    "callbacks": [my_callback],
    "tags": ["prod"],
    "metadata": {"user_id": "123"},
})
```

**`RunnableConfig` fields**:

- `run_id`: Unique identifier for the run (auto-generated)
- `callbacks`: List of callback handlers for tracing/logging
- `tags`: String tags for filtering and grouping in observability tools
- `metadata`: Arbitrary dict for user-defined context
- `run_name`: Human-readable name for the run
- `max_concurrency`: Limit on concurrent operations (for `abatch`)
- `configurable`: Dict of runtime configuration values for `RunnableConfigurableFields` and `RunnableConfigurableAlternatives`

### Configurable Runnable Components

Components can expose configurable parameters as first-class selection points:

```python
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4")

# Make the temperature and model selectable at runtime
configurable_model = model.configurable_fields(
    temperature=ConfigurableField(
        id="temperature",
        name="Temperature",
        description="Randomness of outputs",
        default=0.7,
    )
)

# Select configuration at invocation
result = configurable_model.invoke(
    "...",
    config={"configurable": {"temperature": 0.5}}
)
```

This enables building dynamic pipelines where users can swap components or adjust parameters without rebuilding the chain.

## Extension Points: Creating Custom Runnables

Custom Runnables integrate seamlessly with composition by implementing the protocol:

```python
from langchain_core.runnables import Runnable, RunnableConfig
from typing import Iterator

class LengthCounter(Runnable[str, int]):
    """Custom Runnable that counts character length."""
    
    def invoke(self, input: str, config: RunnableConfig | None = None) -> int:
        return len(input)
    
    async def ainvoke(self, input: str, config: RunnableConfig | None = None) -> int:
        # For I/O, implement actual async logic
        return len(input)
    
    def stream(self, input: str, config: RunnableConfig | None = None) -> Iterator[int]:
        # Stream mode: emit partial results as characters arrive
        count = 0
        for char in input:
            count += 1
            yield count

# Immediately composable
chain = prompt | model | LengthCounter()
```

**Extension pattern guidelines**:

1. **Implement `invoke` (required)**: The core sync method that all other methods delegate to by default.
2. **Implement `ainvoke`** if the operation is I/O-bound (making async native rather than blocking thread pools).
3. **Implement `stream` / `astream`** if the operation naturally produces partial outputs (e.g., token streaming, iterative processing).
4. **Expose `input_schema` and `output_schema`** so composition operators can validate types (automatically derived from generic type parameters or Pydantic models).
5. **Respect `RunnableConfig`**: Extract callbacks, tags, metadata if your component needs to emit lifecycle events or tracing.

## Middleware: Composable Cross-Cutting Concerns

Middleware provides a pattern for layered, composable interception of agent behavior. Each middleware layer wraps the ones beneath, enabling concerns like authentication, caching, retries, and logging without modifying core agent logic.

### Middleware Composition Order

When multiple middleware are registered, **first in the list becomes the outermost layer**:

```python
middleware = [Auth, Cache, Retry]  # Execution order: Auth → Cache → Retry → Agent
```

Each middleware intercepts requests, calls the next layer, receives responses, and can transform them:

```python
# Sync model call stack flow
Auth.wrap_model_call(request, lambda req:
    Cache.wrap_model_call(req, lambda req:
        Retry.wrap_model_call(req, handler(agent))
    )
)
# Request: Auth → Cache → Retry → Model
# Response: Model → Retry → Cache → Auth
```

### Core Lifecycle Hooks

Middleware can implement hooks for different agent phases:

- **`before_agent(state, runtime)`**: Runs once at the start.
- **`before_model(state, runtime)`**: Runs before each model call; can modify state.
- **`wrap_model_call(request, handler)`**: Core interception point; can call handler multiple times (for retries), skip it (caching), or modify the request/response.
- **`after_model(state, runtime)`**: Runs after model response; can approve/reject/modify tool calls.
- **`wrap_tool_call(request, execute)`**: Intercept tool execution (for auth, caching, etc.).
- **`after_agent(state, runtime)`**: Runs once at the end.

Each hook has a sync and async variant (`before_model` / `abefore_model`, etc.).

### Example: Building a Custom Middleware

```python
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse

class LoggingMiddleware(AgentMiddleware):
    """Log all model calls."""
    
    def wrap_model_call(self, request, handler):
        print(f"Calling model with {len(request.messages)} messages")
        try:
            response = handler(request)
            print(f"Model returned: {response}")
            return response
        except Exception as e:
            print(f"Model failed: {e}")
            raise
    
    async def awrap_model_call(self, request, handler):
        print(f"Calling model (async) with {len(request.messages)} messages")
        try:
            response = await handler(request)
            print(f"Model returned: {response}")
            return response
        except Exception as e:
            print(f"Model failed: {e}")
            raise

# Register in agent
from langchain.agents import create_agent

agent = create_agent(
    model="openai:gpt-4o",
    tools=[...],
    middleware=[LoggingMiddleware()],
)
```

### Middleware State Extension

Middleware can extend agent state with custom fields:

```python
from typing import TypedDict, Annotated
from langchain.agents.middleware.types import AgentMiddleware, OmitFromSchema

class RateLimitState(TypedDict):
    rate_limit_remaining: Annotated[int, OmitFromSchema(output=True)]

class RateLimitMiddleware(AgentMiddleware):
    state_schema = RateLimitState
    
    def before_agent(self, state, runtime):
        return {"rate_limit_remaining": 100}
    
    def after_model(self, state, runtime):
        if state.get("rate_limit_remaining", 0) < 10:
            return {"rate_limit_remaining": 100}  # Reset
```

The `OmitFromSchema` annotation hides fields from the input or output schema, useful for internal state that users shouldn't need to provide or see.

## Practical Patterns for Complex Workflows

### Fan-Out / Fan-In: Process and Combine

```python
from langchain_core.runnables import RunnableParallel, RunnableLambda

# Analyze text from multiple angles
analysis = (
    RunnablePassthrough()
    | {
        "sentiment": sentiment_chain,
        "entities": entity_chain,
        "topics": topic_chain,
    }
    | RunnableLambda(lambda results: {
        "analysis": results,
        "confidence": compute_confidence(results),
    })
)
```

### Conditional Error Handling and Retry

```python
from langchain_core.runnables import RunnableBranch, RunnableLambda

resilient_chain = (
    risky_operation
    .with_retry(
        retry_if_exception_type=(TimeoutError, ConnectionError),
        stop_after_attempt=3,
        wait_exponential_jitter=True,
    )
    .with_fallbacks([
        fallback_operation_1,
        fallback_operation_2,
    ])
)
```

### Dynamic Pipeline Composition

```python
from langchain_core.runnables import RunnableConfigurableAlternatives

# Allow users to swap out the model at runtime
model_options = RunnableConfigurableAlternatives(
    default_key="gpt4",
    gpt4=ChatOpenAI(model="gpt-4o"),
    claude=Anthropic(),
    local=OllamaChat(),
)

chain = prompt | model_options | parser

# Invoke with selected model
result = chain.invoke(
    {...},
    config={"configurable": {"which": "claude"}}
)
```

### Nested State Machines with Agents

Agents themselves are Runnables, enabling composition into larger workflows:

```python
from langchain.agents import create_agent

research_agent = create_agent(
    model="openai:gpt-4o",
    tools=[web_search, summarize],
    system_prompt="You are a research assistant.",
)

writer_agent = create_agent(
    model="openai:gpt-4o",
    tools=[write_draft, edit],
    system_prompt="You are a writer.",
)

# Compose agents
workflow = (
    research_agent
    | RunnableLambda(extract_findings)
    | writer_agent
)

# Run the workflow
result = workflow.invoke({"messages": [user_input]})
```

## Observability and Debugging

All composed chains integrate with LangChain's callback and tracing system:

```python
from langchain_core.callbacks import ConsoleCallbackHandler
from langchain_core.globals import set_debug

# Global debug mode
set_debug(True)

# Or per-invocation callbacks
result = chain.invoke(
    input,
    config={
        "callbacks": [ConsoleCallbackHandler()],
        "tags": ["prod"],
    }
)

# Integration with LangSmith for production observability
# Set LANGSMITH_API_KEY environment variable, then:
result = chain.invoke(input)  # Automatically traced to LangSmith
```

## Summary

Composable systems in LangChain emerge from these principles:

1. **Uniform protocol**: Every component implements `Runnable[Input, Output]`, ensuring consistent execution modes (sync, async, batch, stream).
2. **Declarative composition**: The `|` operator, dict syntax, and branching operators enable readable, functional chains.
3. **Type safety**: Generic types and schema inference validate composition correctness without boilerplate.
4. **Dependency injection**: `RunnableConfig` flows through execution, enabling runtime configuration without coupling.
5. **Pluggable extensions**: Middleware, custom Runnables, and configurability patterns enable extending behavior without modifying core logic.
6. **Modularity**: Each component is independently testable; complex workflows are built by combining simpler, proven pieces.

These patterns enable building complex, maintainable, and observable AI applications where components remain loosely coupled and easy to reason about.
