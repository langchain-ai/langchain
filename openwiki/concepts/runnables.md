---
type: Concept
title: "Runnables: Core Execution Abstraction"
description: "Runnable is the foundational execution abstraction in LangChain. It defines a unified, composable interface for synchronous, asynchronous, batch, and streaming execution. Every LangChain component—prompts, models, parsers, retrievers, tools—is a Runnable, enabling declarative composition and uniform control flow."
tags: [runnable, lcel, composition, invoke, stream, batch, async, protocol]
verified:
  - by: openwiki/0.5.0
    at: 2026-10-25T08:23:03.589Z
sources:
  - id: openwiki-source-a1981e868973f6fd7f71e12e
    resource: repo://libs/core/langchain_core/runnables/base.py
  - id: openwiki-source-48e94bbe49ab4f33ba87e9cb
    resource: repo://libs/core/langchain_core/runnables/branch.py
  - id: openwiki-source-079792f059657900794e2955
    resource: repo://libs/core/langchain_core/runnables/config.py
  - id: openwiki-source-f9f4c1dc4f9cdf80d824ce15
    resource: repo://libs/core/langchain_core/runnables/fallbacks.py
  - id: openwiki-source-7a970e5b49bde2fe4f9be3db
    resource: repo://libs/core/langchain_core/runnables/passthrough.py
generated: { by: "openwiki/0.5.0", at: "2026-10-25T08:23:03.589Z" }
---

## Overview

**Runnable** is the foundational execution abstraction in LangChain's core layer. It defines a serializable, composable interface that every LLM component must implement: synchronous invoke (`invoke`), asynchronous invoke (`ainvoke`), batch processing (`batch`, `abatch`), and streaming (`stream`, `astream`). A Runnable is generic over input and output types (`Runnable[Input, Output]`) and manages schema introspection, configuration threading, and callback integration.

The **LangChain Expression Language (LCEL)** leverages Runnables to build chains declaratively. Any chain composed from Runnables automatically inherits sync, async, batch, and streaming support without additional implementation. This unifies execution patterns and enables sophisticated control flows—sequential piping (`|`), parallel forking (dict literals), branching, fallback handling, and retry logic—all composable as first-class operations.

## The Runnable Interface

**Location**: `repo://libs/core/langchain_core/runnables/base.py#L134-L350`

The `Runnable` abstract base class defines the contract for all components. A Runnable is generic over input and output types and must implement only the abstract `invoke` method; all other execution methods have default implementations that subclasses can override for optimization.

### Core Responsibilities

1. **Single invoke**: Execute a single input synchronously and return a single output.
2. **Async invoke**: Execute a single input asynchronously.
3. **Batch processing**: Process multiple inputs in parallel (default uses thread pool; subclasses can optimize for batching-aware backends).
4. **Streaming**: Yield partial outputs as they are produced (default calls invoke once; specialized implementations stream incremental outputs).
5. **Async streaming**: Yield outputs asynchronously (default delegates to sync via executor; specialized implementations use native async).
6. **Schema introspection**: Expose input type, output type, and configuration schema as Pydantic models for validation and tooling.
7. **Composition**: Support chaining with other Runnables via operators and methods.

### Synchronous Execution Methods

**`invoke(input, config=None)`** (`repo://libs/core/langchain_core/runnables/base.py#L886-L907`) is the abstract core method that all subclasses must implement:

```python
@abstractmethod
def invoke(
    self,
    input: Input,
    config: RunnableConfig | None = None,
    **kwargs: Any,
) -> Output:
    """Transform a single input into an output.
    
    Args:
        input: The input to transform.
        config: Configuration for execution (tags, metadata, callbacks, etc.).
    
    Returns:
        The transformed output.
    """
```

- **Must be implemented** by all subclasses.
- Accepts optional `RunnableConfig` for tags, metadata, callbacks, recursion limits, and configurable parameters.
- Returns a single output.

**`batch(inputs, config=None, return_exceptions=False)`** (`repo://libs/core/langchain_core/runnables/base.py#L932-L1010`) processes multiple inputs:

```python
def batch(
    self,
    inputs: list[Input],
    config: RunnableConfig | list[RunnableConfig] | None = None,
    *,
    return_exceptions: bool = False,
    **kwargs: Any | None,
) -> list[Output]:
    """Process multiple inputs in parallel."""
```

- **Default**: Calls `invoke` for each input in parallel via `ThreadPoolExecutor`.
- Accepts a single config (applied to all) or a list of configs (one per input).
- If `return_exceptions=True`, exceptions are returned as-is in the output list; otherwise, first exception is raised.
- Subclasses override for batching-aware backends (e.g., LLMs with batch API support).

**`stream(input, config=None)`** (`repo://libs/core/langchain_core/runnables/base.py#L1195-L1213`) yields partial outputs:

```python
def stream(
    self,
    input: Input,
    config: RunnableConfig | None = None,
    **kwargs: Any | None,
) -> Iterator[Output]:
    """Yield outputs as they are produced."""
```

- **Default**: Yields one full output from `invoke`.
- **Specialized implementations**: Chat models and streaming parsers override to yield partial outputs (e.g., tokens) as they arrive.
- **Transform method**: Advanced streaming is implemented via `transform(input_iterator)`, which maps a stream of inputs to outputs without blocking. This enables progressive end-to-end streaming through entire chains.

### Asynchronous Execution Methods

**`ainvoke(input, config=None)`** (`repo://libs/core/langchain_core/runnables/base.py#L909-L930`) is the async variant:

```python
async def ainvoke(
    self,
    input: Input,
    config: RunnableConfig | None = None,
    **kwargs: Any,
) -> Output:
    """Transform a single input into an output asynchronously."""
```

- **Default**: Delegates to sync `invoke` via `run_in_executor`.
- **Subclasses override** for native async (e.g., async API calls).

**`abatch(inputs, config=None, return_exceptions=False)`** (`repo://libs/core/langchain_core/runnables/base.py#L1066-L1112`) batches asynchronously:

```python
async def abatch(
    self,
    inputs: list[Input],
    config: RunnableConfig | list[RunnableConfig] | None = None,
    *,
    return_exceptions: bool = False,
    **kwargs: Any | None,
) -> list[Output]:
    """Process multiple inputs concurrently."""
```

- Calls `ainvoke` for each input concurrently via `asyncio.gather`.
- Respects `max_concurrency` from config to limit parallelism.

**`astream(input, config=None)`** (`repo://libs/core/langchain_core/runnables/base.py#L1216-L1235`) is the async streaming variant:

```python
async def astream(
    self,
    input: Input,
    config: RunnableConfig | None = None,
    **kwargs: Any | None,
) -> AsyncIterator[Output]:
    """Yield outputs asynchronously as they are produced."""
```

- **Default**: Yields one output from `ainvoke`.
- **Specialized** for streaming backends.

### Schema Introspection

**`input_schema` / `output_schema`** (`repo://libs/core/langchain_core/runnables/base.py#L374-L527`) expose input and output types as Pydantic models:

```python
@property
def input_schema(self) -> TypeBaseModel:
    """The type of input this Runnable accepts specified as a Pydantic model."""
    return self.get_input_schema()

@property
def output_schema(self) -> TypeBaseModel:
    """The type of output this Runnable produces specified as a Pydantic model."""
    return self.get_output_schema()
```

- Inferred from generic type parameters or implementer-provided type hints.
- Can be converted to JSON Schema for API documentation, validation, and tool calling.

**`config_schema(include=None)`** returns a Pydantic model for configuration fields marked as configurable via `configurable_fields()` or `configurable_alternatives()`.

## Composition: Piping and Parallel Operators

The heart of LCEL is declarative composition. Runnables are chained together using operators and methods that create new composite Runnables without mutating the original.

### Sequential Composition: Pipe Operator `|`

**`__or__(other)` / `__ror__(other)`** (`repo://libs/core/langchain_core/runnables/base.py#L648-L722`) creates a `RunnableSequence`:

```python
def __or__(self, other):
    """Compose this Runnable with another to create RunnableSequence."""
    return RunnableSequence(self, coerce_to_runnable(other))
```

**Example**:
```python
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

prompt = PromptTemplate.from_template("Tell me about {topic}")
model = ChatOpenAI()
parser = StrOutputParser()

chain = prompt | model | parser
result = chain.invoke({"topic": "machine learning"})
# result is a string, the parsed model output
```

- `other` can be another `Runnable`, a callable, a dict (coerced to `RunnableParallel`), or any `RunnableLike`.
- Automatically flattens nested `RunnableSequence` for efficiency.
- The output of the left side becomes the input to the right side.

**`pipe(*others, name=None)`** (`repo://libs/core/langchain_core/runnables/base.py#L724-L771`) is the explicit method form:

```python
sequence = runnable_1.pipe(runnable_2, runnable_3)
# Equivalent to: runnable_1 | runnable_2 | runnable_3
```

### Parallel Composition: Fork with Dict

**Dict literal** or **`RunnableParallel`** (`repo://libs/core/langchain_core/runnables/base.py#L3868-L3950`) invokes multiple Runnables concurrently with the same input:

```python
from langchain_core.runnables import RunnableParallel

# Via dict literal in a sequence
chain = prompt | {
    "branch_a": model_a,
    "branch_b": model_b,
}

# Explicit RunnableParallel
parallel = RunnableParallel(
    result1=runnable_1,
    result2=runnable_2,
)
```

- All runnables receive the same input.
- Results are collected into a dict with user-specified keys.
- Execution is parallel via `asyncio.gather` or thread pool.
- Useful for processing branches in parallel: multi-chain retrieval, multi-aspect analysis, etc.

**Key behavior**:
- Sync `invoke` uses thread pool executor.
- Async `ainvoke` uses `asyncio.gather` with concurrency control from `max_concurrency` config.
- Both methods preserve exceptions unless `return_exceptions=True`.

## Sequential Composition: RunnableSequence

**Location**: `repo://libs/core/langchain_core/runnables/base.py#L3079-L3235`

`RunnableSequence` is the composition engine for sequential execution. It chains multiple `Runnable` objects where each output feeds into the next input. The `first`, `middle`, and `last` attributes store the steps; the sequence automatically optimizes batch and stream operations by calling each step's batch/stream method in order.

```python
from langchain_core.runnables import RunnableSequence

sequence = RunnableSequence(
    first=prompt,
    middle=[some_runnable],
    last=parser,
)

# Equivalent to: prompt | some_runnable | parser
```

**Key behavior**:

- **Invocation**: `invoke` calls each step's `invoke` method sequentially, passing the output of one as input to the next. Config is merged and propagated.
- **Batching**: Each step in the sequence is called with the batch of inputs from the previous step. If a step's batch is optimized (e.g., for an LLM API), the entire chain benefits.
- **Streaming**: If all steps implement `transform` (streaming input → streaming output), the sequence chains them end-to-end, yielding outputs progressively. If any step lacks `transform`, streaming begins after that blocking step.
- **Async**: Async variants call each step's async method in order via `asyncio.gather` for concurrency control.
- **Flattening**: Nested sequences are flattened to avoid redundant nesting.

## Branching and Routing

### Branching: RunnableBranch

**Location**: `repo://libs/core/langchain_core/runnables/branch.py#L43-L290`

`RunnableBranch` selects and runs one of several branches based on condition evaluation:

```python
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: isinstance(x, str), lambda x: x.upper()),
    (lambda x: isinstance(x, int), lambda x: x + 1),
    lambda x: "default",  # fallback if no condition matches
)

branch.invoke("hello")  # "HELLO"
branch.invoke(42)       # 43
branch.invoke(None)     # "default"
```

- **Constructor**: Accepts variable args of `(condition, runnable)` tuples plus a final default runnable. Each is coerced via `coerce_to_runnable()`.
- **Conditions**: Are themselves Runnables (or coerced from callables returning `bool`). Can be sync, async, or awaitable.
- **Evaluation**: At invoke time, conditions are evaluated sequentially against the same input. The first condition returning `True` selects its runnable.
- **Default**: If no condition matches, the default runnable is executed.
- **Async support**: Both `ainvoke()` and sync `invoke()` support async conditions and runnables natively.
- **Streaming**: Delegates to the selected branch's `stream()` or `astream()` method.
- **Validation**: Enforces minimum 2 branches at construction time.
- **Callback integration**: Tags callback runs as `condition:N` for condition evaluation and `branch:N` or `branch:default` for runnable execution.

### Routing: RouterRunnable

**Location**: `repo://libs/core/langchain_core/runnables/router.py#L46-L150`

`RouterRunnable` routes to a runnable by key:

```python
from langchain_core.runnables import RouterRunnable

router = RouterRunnable(runnables={
    "math": math_chain,
    "text": text_chain,
})

router.invoke({"key": "math", "input": "2 + 2"})  # Uses math_chain
```

- Input is a dict with `"key"` (string identifying the route) and `"input"` (the actual data).
- The selected runnable processes the input.
- Useful for dispatch tables and multi-expert architectures.

## Fallback and Retry Patterns

### Fallbacks: RunnableWithFallbacks

**Location**: `repo://libs/core/langchain_core/runnables/fallbacks.py#L37-L590`

```python
from langchain_core.runnables import RunnableWithFallbacks

model = ChatOpenAI().with_fallbacks([ChatAnthropic(), ChatCohere()])
# Try ChatOpenAI first, then ChatAnthropic, then ChatCohere if prior ones fail.

result = model.invoke("Hello")  # Returns first successful result
```

**Key behavior**:

- **Sequential execution**: Invokes the primary `runnable`, then each fallback in order until one succeeds.
- **Exception handling**: Catches exceptions matching `exceptions_to_handle` (default `Exception`) and proceeds to the next fallback. Exceptions outside this set are raised immediately.
- **Exception injection**: If `exception_key` is set (a string), handled exceptions are injected into the input dict before invoking the next fallback, enabling adaptive recovery strategies.
- **Async support**: Both `ainvoke()` and `abatch()` are natively async, awaiting each fallback in sequence.
- **Batch handling**: Processes batches by invoking each fallback's `batch()` method on all inputs, tracking per-input success/failure to determine which to retry.
- **Callback integration**: Tags each fallback invocation with callbacks for observability; first error and last error are tracked and reported.
- **Streaming**: Delegates to the first successful runnable's `stream()` or `astream()`.
- **Fallback exhaustion**: If all runnables fail, raises the first error encountered.

### Retry: with_retry()

Retry logic is available via the `with_retry()` method on any Runnable:

```python
from langchain_openai import ChatOpenAI
from langchain_core.exceptions import APIError

runnable = ChatOpenAI().with_retry(
    retry_if_exception_type=(APIError,),
    stop_after_attempt=3,
    wait_exponential_jitter=True,
)
# Retries on APIError up to 3 times with exponential backoff + jitter.
```

**Key behavior**:

- Wraps a `Runnable` to add retry logic powered by `tenacity`.
- **Exception filtering**: `retry_if_exception_type` is either a tuple of exception types or a callable that returns `True` for exceptions to retry.
- **Retry strategy**: Configurable via `stop_after_attempt` (default 3) and `wait_exponential_jitter` (default `True`) for exponential backoff with random jitter.
- **Jitter parameters**: Fine-tunable via exponential jitter configuration (initial, max, exp_base, jitter).
- **Retry propagation**: Tracks per-attempt state to tag callback runs as `retry:attempt:N` for observability.
- **Batch retries**: Retries only failed inputs in a batch, re-invoking just the subset that did not succeed.
- **Stream limitation**: `stream()` and `transform()` are not retried because stream retry semantics are ambiguous.
- Best applied to individual runnables (e.g., LLM calls) rather than entire chains.

## Configuration Threading: RunnableConfig

**Location**: `repo://libs/core/langchain_core/runnables/config.py#L57-L129`

`RunnableConfig` is a `TypedDict` that carries execution context through the chain:

```python
class RunnableConfig(TypedDict, total=False):
    tags: list[str]              # For filtering runs, grouping telemetry
    metadata: dict[str, Any]     # Arbitrary metadata (JSON-serializable)
    callbacks: Callbacks         # Lifecycle handlers (on_start, on_end, on_error, etc.)
    run_name: str                # Name for tracing/logging
    max_concurrency: int | None  # Limit parallel execution (for batch/stream concurrency)
    recursion_limit: int         # Prevent infinite recursion (default 25)
    configurable: dict[str, Any] # Runtime config overrides for configurable fields
    run_id: uuid.UUID | None     # Unique execution ID
```

### Config Propagation and Merging

- **Propagation**: Config is threaded through child runnables via context variables (`var_child_runnable_config`) and explicit parameter passing. When a Runnable invokes child runnables, configs are merged (tags and metadata accumulate).
- **Merging**: `merge_configs()` combines parent and child configs, with child values taking precedence. Lists like `tags` are concatenated.
- **Callbacks**: Attached callbacks receive hooks for all intermediate steps, enabling observability and custom logic.
- **Configurable fields**: The `configurable` dict supplies runtime values for fields marked with `configurable_fields()` or `configurable_alternatives()`, enabling dynamic behavior without reconstruction.
- **Context variables**: `var_child_runnable_config` is an `asyncio.ContextVar` that automatically passes config down the call stack for child runnables without explicit parameter passing.

### Configurable Fields

**`configurable_fields(**fields)`** marks fields as runtime-configurable:

```python
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-4"
).configurable_fields(
    model=ConfigurableField(
        id="model_name",
        name="Model Name",
        description="The model to use",
    )
)

# At runtime, override the model:
result = model.invoke(
    "Hello",
    config={"configurable": {"model_name": "gpt-3.5-turbo"}}
)
```

- Fields are exposed in `config_schema()`.
- Runtime values are applied before invoke.

## Streaming and Event Flow

Streaming is not an afterthought in Runnables; it is a first-class execution mode. The streaming protocol enables responsive, incremental output generation.

### Streaming Semantics

When `stream(input)` is called:

1. **Default behavior**: If a Runnable does not override `stream()`, it yields a single output from `invoke()`.
2. **Streaming models**: Chat models and other components override `stream()` to yield partial outputs (e.g., tokens) as they arrive.
3. **Transform method**: Advanced streaming is implemented via `transform(input_iterator)`, which maps a stream of inputs to outputs without blocking. This enables progressive end-to-end streaming through entire chains.

### RunnableSequence Streaming Optimization

`RunnableSequence` optimizes streaming by analyzing each step:

- If all steps implement `transform`, the sequence chains them: input flows through each step's transform, yielding outputs progressively.
- If any step lacks `transform`, streaming begins after that step. Inputs before it are processed in full, then partial outputs flow through remaining steps.
- This ensures end-to-end streaming when possible and graceful degradation when not.

**Example**:
```python
# All steps stream-compatible: progressive token output
chain = prompt | streaming_model | streaming_parser
for chunk in chain.stream({"topic": "AI"}):
    print(chunk, end="", flush=True)

# Parser lacks transform: tokens blocked until parsing
chain = prompt | streaming_model | slow_parser
# Streaming begins after model completes and parser starts
```

## Core Runnable Types

### RunnableLambda

**Location**: `repo://libs/core/langchain_core/runnables/base.py#L4707-L4850`

`RunnableLambda` wraps a Python callable into a `Runnable`:

```python
from langchain_core.runnables import RunnableLambda

def add_one(x: int) -> int:
    return x + 1

runnable = RunnableLambda(add_one)
runnable.invoke(1)  # 2

# Async support
async def add_one_async(x: int) -> int:
    return x + 1

runnable = RunnableLambda(add_one, afunc=add_one_async)
await runnable.ainvoke(1)  # 2
```

**Key behavior**:

- Ideal for wrapping custom logic, data transformations, and simple operations.
- If a lambda returns a `Runnable`, that runnable is automatically invoked.
- Does not support streaming by default (use `RunnableGenerator` for streaming lambdas).
- Automatically detects async callables and provides native async support.
- Infers input/output types from function signature for schema introspection.

### RunnableGenerator

Wraps generator functions (sync or async) to create streaming runnables:

```python
from langchain_core.runnables import RunnableGenerator

def token_stream(input: str) -> Iterator[str]:
    for token in input.split():
        yield token

runnable = RunnableGenerator(token_stream)
for chunk in runnable.stream("hello world"):
    print(chunk)  # Yields 'hello', then 'world'
```

**Key behavior**:

- Useful for implementing custom streaming logic.
- Supports both sync generators and async generators.
- Automatically implements `transform` for end-to-end streaming.

### RunnablePassthrough

**Location**: `repo://libs/core/langchain_core/runnables/passthrough.py`

Passes input through unchanged, useful in parallel branches to preserve input for later steps:

```python
from langchain_core.runnables import RunnablePassthrough

chain = prompt | {
    "original_input": RunnablePassthrough(),
    "model_output": model,
}
# Output: {"original_input": <input>, "model_output": <model result>}
```

### RunnablePick

Selects specific keys from dict output:

```python
chain | RunnablePick("key_a")  # Output only "key_a"
# Or: chain.pick(["key_a", "key_b"])
```

### RunnableAssign

Adds new fields to dict output by invoking additional runnables:

```python
chain.assign(new_field=some_runnable)
# Output now includes original fields + new_field
```

## RunnableLike and Type Coercion

**Location**: `repo://libs/core/langchain_core/runnables/base.py#L6612-L6668`

`RunnableLike` is a union type that accepts anything composable:

```python
RunnableLike = (
    Runnable[Input, Output]
    | Callable[[Input], Output]
    | Callable[[Input], Awaitable[Output]]
    | Callable[[Iterator[Input]], Iterator[Output]]
    | Callable[[AsyncIterator[Input]], AsyncIterator[Output]]
    | Mapping[str, Any]
)
```

**`coerce_to_runnable(thing)`** converts any `RunnableLike` into a `Runnable`:

```python
def coerce_to_runnable(thing: RunnableLike) -> Runnable:
    """Coerce a Runnable-like object into a Runnable."""
    if isinstance(thing, Runnable):
        return thing  # Already a Runnable
    if is_async_generator(thing) or inspect.isgeneratorfunction(thing):
        return RunnableGenerator(thing)  # Wrap generators
    if callable(thing):
        return RunnableLambda(thing)  # Wrap functions
    if isinstance(thing, dict):
        return RunnableParallel(thing)  # Coerce dicts to parallel
    raise TypeError("...")  # Unsupported type
```

This enables the intuitive syntax: `prompt | my_function | {"field": another_function}`. Each element is automatically coerced to a Runnable.

## Callbacks and Observability

Runnables integrate with the callback system via `RunnableConfig`:

```python
from langchain_core.tracers import ConsoleCallbackHandler

chain.invoke(
    input,
    config={
        "callbacks": [ConsoleCallbackHandler()],
        "tags": ["production", "query"],
        "metadata": {"user_id": 123},
    }
)
```

**Callback hooks**:
- `on_runnable_start`: When a step begins
- `on_runnable_end`: When a step completes successfully
- `on_runnable_error`: When a step fails
- `on_llm_new_token`: Token-by-token streaming
- And many others

This enables real-time monitoring, custom logging, user attribution, and performance tracking. All composition operators (sequence, parallel, branch, fallback, retry) automatically integrate with callbacks, enabling end-to-end tracing of complex chains.

## Key Design Invariants

### Immutability and Fluent API

Composition methods (`with_fallbacks`, `with_retry`, `pick`, `assign`, `pipe`, etc.) return new `Runnable` instances; they do not mutate the original. This enables safe chaining and allows reuse of components without side effects.

### Type Transparency

Input and output types are exposed and enforced:
- `input_schema` validates input before invoke
- `output_schema` documents expected output for tools and UIs
- JSON schema generation enables API/OpenAPI documentation

### Lazy Execution

Chains are constructed lazily—composing with `|` does not execute anything. Execution happens only on `invoke`, `ainvoke`, `batch`, `abatch`, `stream`, or `astream`.

### Streaming as a First-Class Concern

Streaming is not an afterthought; it is a core execution mode. Every Runnable exposes `stream` and `astream`, enabling responsive and incremental output. The `transform` method enables true progressive streaming through chains.

### Concurrency and Parallelism

Batch operations use thread pools or asyncio concurrency to process inputs in parallel. Config controls parallelism via `max_concurrency`. Async methods are first-class, enabling high-concurrency server deployments.

### Uniformity Across Execution Modes

All Runnables implement the same interface:
- Sync and async variants exist for all operations.
- Single-input and batch operations have matching semantics.
- Streaming follows the same composition patterns as non-streaming.

This uniformity enables a single chain to be used in sync, async, batch, and streaming contexts without modification.

## Practical Example: Chat Chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel

# Define output schema
class Summary(BaseModel):
    title: str
    key_points: list[str]

# Create the chain
prompt = ChatPromptTemplate.from_template(
    "Summarize the following text in JSON format:\n{text}"
)
model = ChatOpenAI(model="gpt-4")
parser = JsonOutputParser(pydantic_object=Summary)

chain = prompt | model | parser

# Sync invoke
result = chain.invoke({"text": "..."})
# result is a Summary instance

# Async invoke
import asyncio
result = asyncio.run(chain.ainvoke({"text": "..."}))

# Batch processing
results = chain.batch([
    {"text": "text1"},
    {"text": "text2"},
])

# Streaming
for chunk in chain.stream({"text": "..."}):
    print(chunk)
```

All execution modes are available on the same chain without modification, demonstrating the power of the Runnable abstraction.

## All Components Are Runnables

A key design principle: **everything is a Runnable**. This includes:

- **Prompts** (`PromptTemplate`, `ChatPromptTemplate`): Format input variables into messages.
- **Chat Models** (`ChatOpenAI`, `ChatAnthropic`): Invoke LLM APIs.
- **Output Parsers** (`JsonOutputParser`, `StrOutputParser`): Parse model output into structured types.
- **Retrievers** (`VectorStoreRetriever`, `BM25Retriever`): Fetch relevant documents.
- **Tools** (`BaseTool`): Callable functions with schemas.
- **Chains**: Composite runnables built from other runnables.
- **Agents**: Orchestrate tools and runnables in feedback loops.

Because all are Runnables, any can be composed with any other via `|`, parallelized, retried, and monitored uniformly. This universal interface eliminates friction and enables elegant, type-safe composition.
