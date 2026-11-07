---
type: "API Reference"
title: "Runnables: Complete Protocol Reference"
description: "Complete API reference for the Runnable protocol, execution methods, composition operators, configuration threading, and streaming patterns in LangChain."
tags: [runnable, lcel, protocol, api, reference, invoke, batch, stream, composition]
verified:
  - by: openwiki/0.5.0
    at: 2026-11-07T08:15:07.286Z
sources:
  - id: openwiki-source-a1981e868973f6fd7f71e12e
    resource: repo://libs/core/langchain_core/runnables/base.py
  - id: openwiki-source-48e94bbe49ab4f33ba87e9cb
    resource: repo://libs/core/langchain_core/runnables/branch.py
  - id: openwiki-source-079792f059657900794e2955
    resource: repo://libs/core/langchain_core/runnables/config.py
  - id: openwiki-source-f9f4c1dc4f9cdf80d824ce15
    resource: repo://libs/core/langchain_core/runnables/fallbacks.py
generated: { by: "openwiki/0.5.0", at: "2026-11-07T08:15:07.286Z" }
---

# Runnables: Complete Protocol Reference

This is the complete API reference for the `Runnable` protocol, composition operators, configuration, and execution semantics in LangChain Core. For conceptual overview and design principles, see [Runnables: Core Execution Abstraction](/openwiki/concepts/runnables.md).

## The Runnable Protocol

**Location**: `repo://libs/core/langchain_core/runnables/base.py#L197-L350`

`Runnable` is an abstract base class that defines the unified execution interface for all LangChain components. It is generic over input and output types: `Runnable[Input, Output]`.

### Abstract Method: invoke

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
        config: Optional configuration for execution.
    
    Returns:
        The transformed output.
    """
```

This is the only abstract method. All other execution methods have default implementations that subclasses can override for optimization.

## Execution Methods

### Synchronous Execution

**`invoke(input, config=None)`** (`repo://libs/core/langchain_core/runnables/base.py#L949-L971`)

Transforms a single input synchronously.

**Arguments**:
- `input` (Input): The input to process.
- `config` (RunnableConfig | None): Configuration for execution including tags, metadata, callbacks, max concurrency, and configurable parameter overrides.

**Returns**: The transformed output.

**`batch(inputs, config=None, *, return_exceptions=False)`** (`repo://libs/core/langchain_core/runnables/base.py#L995-L1044`)

Processes multiple inputs in parallel.

**Arguments**:
- `inputs` (list[Input]): List of inputs to process.
- `config` (RunnableConfig | list[RunnableConfig] | None): A single config applied to all inputs, a list of configs (one per input), or None.
- `return_exceptions` (bool): If True, exceptions are returned as-is in the output list instead of raising the first exception.

**Returns**: `list[Output]` - A list of outputs.

**Default implementation**: Calls `invoke` for each input in parallel using `ThreadPoolExecutor`. Subclasses override for batching-aware backends (e.g., LLM batch APIs).

**`batch_as_completed(inputs, config=None, *, return_exceptions=False)`** (`repo://libs/core/langchain_core/runnables/base.py#L1046-L1128`)

Processes multiple inputs in parallel, yielding results as they complete.

**Arguments**: Same as `batch`.

**Yields**: `tuple[int, Output]` - Index and result tuples as they complete, enabling streaming processing of batch results.

**`stream(input, config=None)`** (`repo://libs/core/langchain_core/runnables/base.py#L1258-L1278`)

Yields partial outputs as they are produced.

**Arguments**:
- `input` (Input): Single input to stream.
- `config` (RunnableConfig | None): Execution configuration.

**Yields**: Output chunks progressively.

**Default implementation**: Yields a single complete output from `invoke()`. Streaming components (chat models, parsers) override to yield incremental outputs (e.g., tokens).

**`transform(input, config=None)`** (`repo://libs/core/langchain_core/runnables/base.py#L1824-L1862`)

Maps an iterator of inputs to an iterator of outputs without blocking on full input.

**Arguments**:
- `input` (Iterator[Input]): Iterator of inputs.
- `config` (RunnableConfig | None): Execution configuration.

**Yields**: Output chunks progressively.

**Default implementation**: Buffers all input using `+` operator, then calls `stream()`. Streaming components override to process input incrementally, enabling end-to-end progressive streaming through chains.

### Asynchronous Execution

**`ainvoke(input, config=None)`** (`repo://libs/core/langchain_core/runnables/base.py#L972-L994`)

Transforms a single input asynchronously.

**Default implementation**: Delegates to `invoke()` via `run_in_executor()`. Subclasses override for native async (API calls, database queries).

**`abatch(inputs, config=None, *, return_exceptions=False)`** (`repo://libs/core/langchain_core/runnables/base.py#L1130-L1177`)

Processes multiple inputs concurrently using asyncio.

**Default implementation**: Calls `ainvoke()` for each input concurrently via `asyncio.gather()`. Respects `max_concurrency` from config to limit parallelism.

**`abatch_as_completed(inputs, config=None, *, return_exceptions=False)`** (`repo://libs/core/langchain_core/runnables/base.py#L1178-L1256`)

Processes multiple inputs concurrently, yielding results as they complete.

**Yields**: `tuple[int, Output]` - Index and result tuples as they complete.

**`astream(input, config=None)`** (`repo://libs/core/langchain_core/runnables/base.py#L1279-L1299`)

Yields outputs asynchronously as they are produced.

**Default implementation**: Yields a single output from `ainvoke()`. Streaming components override for native async streaming.

**`atransform(input, config=None)`** (`repo://libs/core/langchain_core/runnables/base.py#L1863-L1902`)

Maps an async iterator of inputs to an async iterator of outputs without blocking.

**Default implementation**: Buffers all input, then calls `astream()`. Streaming components override for incremental async processing.

### Streaming and Event Methods

**`astream_log(input, config=None, *, diff=True, with_streamed_output_list=True, include_names=None, include_types=None, include_tags=None, exclude_names=None, exclude_types=None, exclude_tags=None)`** (`repo://libs/core/langchain_core/runnables/base.py#L1301-L1405`)

Streams all outputs and selected intermediate results as structured logs.

**Arguments**:
- `input` (Any): The input to execute.
- `diff` (bool): If True, yield JSON patch diffs between steps; if False, yield complete state.
- `with_streamed_output_list` (bool): Include the `streamed_output` list in logs.
- `include_names/types/tags`: Filter logs by name, type, or tag (include these).
- `exclude_names/types/tags`: Filter logs by name, type, or tag (exclude these).

**Yields**: `RunLogPatch | RunLog` - Structured logs with JSON patch operations or complete state.

**Note**: Deprecated as of 1.3.3. Use `astream_events()` instead.

**`astream_events(input, config=None, *, version="v2", include_names=None, include_types=None, include_tags=None, exclude_names=None, exclude_types=None, exclude_tags=None)`** (`repo://libs/core/langchain_core/runnables/base.py#L1406-L1736`)

Streams real-time events from chain execution, including intermediate results and custom events.

**Arguments**:
- `input` (Any): The input to execute.
- `version` ("v1" | "v2" | "v3"): Event schema version. Default "v2" is recommended.
  - "v1": Deprecated; use "v2".
  - "v2": Yields typed `StreamEvent` dicts; supports custom events.
  - "v3": Content-block-centric protocol for chat models and compiled graphs (beta).
- Filter arguments select events by name, type, or tag.

**Yields**: `StreamEvent` - Dictionary events with schema:
- `event` (str): Event name format `on_[component_type]_(start|stream|end)`.
- `name` (str): Name of the Runnable that generated the event.
- `run_id` (str): Unique ID for this execution.
- `parent_ids` (list[str]): IDs of parent runnables (v2+ only).
- `tags` (list[str]): Tags associated with the Runnable.
- `metadata` (dict): Metadata associated with the Runnable.
- `data` (dict): Event-specific data (input, output, chunk, etc.).

**Raises**: `NotImplementedError` if version="v3" on a Runnable that doesn't implement v3 protocol.

**`stream_events(input, config=None, *, version="v2", ...)`** 

Synchronous variant of `astream_events`. Returns a generator instead of an async iterator.

## Schema Introspection

**`InputType`** property

The Python type annotation for input, inferred from generic parameters.

**`OutputType`** property

The Python type annotation for output, inferred from generic parameters.

**`input_schema`** property (`repo://libs/core/langchain_core/runnables/base.py#L439-L442`)

Pydantic model representing input structure. Used for validation and tool calling.

```python
@property
def input_schema(self) -> TypeBaseModel:
    """The Pydantic model for this Runnable's input."""
```

**`output_schema`** property (`repo://libs/core/langchain_core/runnables/base.py#L514-L521`)

Pydantic model representing output structure.

```python
@property
def output_schema(self) -> TypeBaseModel:
    """The Pydantic model for this Runnable's output."""
```

**`get_input_jsonschema(config=None)`** (`repo://libs/core/langchain_core/runnables/base.py#L484-L513`)

Returns JSON Schema representation of input for API documentation and OpenAPI.

**`get_output_jsonschema(config=None)`**

Returns JSON Schema representation of output.

**`config_schema(include=None)`**

Returns Pydantic model for configuration fields marked as configurable via `configurable_fields()` or `configurable_alternatives()`.

## Composition Operators

Composition creates new Runnables from existing ones without mutation.

### Sequential Composition: Pipe Operator

**`__or__(other)` / `__ror__(other)`** (`repo://libs/core/langchain_core/runnables/base.py#L648-L722`)

Creates a `RunnableSequence` where output of left side feeds into right side.

```python
chain = prompt | model | parser
```

**Arguments**:
- `other`: Can be a `Runnable`, callable, dict (coerced to `RunnableParallel`), or any `RunnableLike`.

**Returns**: `RunnableSequence` - automatically flattens nested sequences.

**`pipe(*others, name=None)`** (`repo://libs/core/langchain_core/runnables/base.py#L724-L771`)

Explicit method form of `|` operator.

```python
sequence = runnable_1.pipe(runnable_2, runnable_3)
```

### Parallel Composition

**Dict literal or `RunnableParallel`** (`repo://libs/core/langchain_core/runnables/base.py#L3868-L3950`)

Invokes multiple Runnables with the same input, collecting results into a dict.

```python
parallel = prompt | {
    "branch_a": model_a,
    "branch_b": model_b,
}
```

**Behavior**:
- All runnables receive the same input.
- Results collected into a dict with user-specified keys.
- Execution is parallel via `asyncio.gather()` (async) or thread pool (sync).
- Respects `max_concurrency` config.

### RunnableSequence

**Location**: `repo://libs/core/langchain_core/runnables/base.py#L3079-L3235`

`RunnableSequence` chains multiple Runnables sequentially.

**Constructor**:
```python
sequence = RunnableSequence(first=r1, middle=[r2, r3], last=r4)
# Equivalent to: r1 | r2 | r3 | r4
```

**Key behavior**:
- **invoke**: Sequentially calls each step's invoke, passing output to next input.
- **batch**: Each step batches its inputs from previous step's outputs. Optimizations propagate (e.g., LLM batch API).
- **stream**: If all steps implement `transform`, chains them end-to-end for progressive output. If any step lacks `transform`, streams begin after that step.
- **Flattening**: Nested sequences automatically flatten.

## Branching and Routing

### RunnableBranch

**Location**: `repo://libs/core/langchain_core/runnables/branch.py#L43-L290`

Selects one branch based on condition evaluation.

```python
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: isinstance(x, str), lambda x: x.upper()),
    (lambda x: isinstance(x, int), lambda x: x + 1),
    lambda x: "default",
)

branch.invoke("hello")  # "HELLO"
```

**Constructor**: Variable args of `(condition, runnable)` tuples plus final default runnable.

**Behavior**:
- **Conditions**: Evaluated sequentially against input; first matching condition wins.
- **Coercion**: All arguments coerced via `coerce_to_runnable()`.
- **Async support**: Conditions and runnables can be async; both `invoke()` and `ainvoke()` work natively.
- **Streaming**: Delegates to selected branch's `stream()` or `astream()`.
- **Callback tags**: Conditions tagged as `condition:N`, branches as `branch:N` or `branch:default`.
- **Validation**: Requires minimum 2 branches.

### RouterRunnable

**Location**: `repo://libs/core/langchain_core/runnables/router.py#L46-L150`

Routes to a runnable by key.

```python
router = RouterRunnable(runnables={
    "math": math_chain,
    "text": text_chain,
})

router.invoke({"key": "math", "input": "2 + 2"})
```

**Input schema**: Dict with required keys `"key"` (route selection) and `"input"` (data).

## Fallback and Retry

### RunnableWithFallbacks

**Location**: `repo://libs/core/langchain_core/runnables/fallbacks.py#L37-L590`

Executes primary runnable, then fallbacks in sequence until one succeeds.

```python
model = ChatOpenAI().with_fallbacks([ChatAnthropic(), ChatCohere()])
result = model.invoke("Hello")  # First successful result
```

**Behavior**:
- **Sequential execution**: Primary first, then each fallback in order.
- **Exception handling**: Catches exceptions matching `exceptions_to_handle` (default `Exception`); other exceptions raise immediately.
- **Exception injection**: If `exception_key` is set, handled exceptions are injected into the input dict before next fallback.
- **Batch handling**: Tracks per-input success/failure; retries only failed inputs on next fallback.
- **Streaming**: Delegates to first successful runnable's `stream()` or `astream()`.
- **Exhaustion**: If all fail, raises first error.

### RunnableRetry (via with_retry)

**Location**: `repo://libs/core/langchain_core/runnables/retry.py`

Retries a Runnable on exception using exponential backoff.

```python
runnable = model.with_retry(
    retry_if_exception_type=(APIError,),
    stop_after_attempt=3,
    wait_exponential_jitter=True,
)
```

**Arguments**:
- `retry_if_exception_type`: Tuple of exception types or callable predicate.
- `stop_after_attempt`: Max retry attempts (default 3).
- `wait_exponential_jitter`: Add jitter to exponential backoff (default True).
- `exponential_jitter_params`: Fine-tune jitter (initial, max, exp_base, jitter).

**Behavior**:
- **Batch retries**: Retries only failed inputs.
- **Stream limitation**: `stream()` and `transform()` are not retried (ambiguous semantics).
- **Callback tagging**: Attempts tagged as `retry:attempt:N`.
- **Best practice**: Apply to individual components (e.g., LLM calls) rather than entire chains.

## Configuration Threading: RunnableConfig

**Location**: `repo://libs/core/langchain_core/runnables/config.py#L57-L129`

`RunnableConfig` is a `TypedDict` carrying execution context through the chain.

```python
class RunnableConfig(TypedDict, total=False):
    tags: list[str]                    # Filtering and grouping telemetry
    metadata: dict[str, Any]           # Arbitrary metadata (JSON-serializable)
    callbacks: Callbacks               # Lifecycle handlers
    run_name: str                      # Name for tracing/logging
    max_concurrency: int | None        # Limit parallel execution
    recursion_limit: int               # Prevent infinite recursion (default 25)
    configurable: dict[str, Any]       # Runtime parameter overrides
    run_id: uuid.UUID | None           # Unique execution ID
```

### Config Propagation

- **Threading**: Passed explicitly to all execution methods; also threaded via context variables (`var_child_runnable_config`) for implicit child access.
- **Merging**: `merge_configs()` combines parent and child configs; child values take precedence; lists like `tags` concatenate.
- **Callbacks**: All intermediate steps call registered callbacks; enables observability without modifying chain code.
- **Configurable fields**: The `configurable` dict supplies runtime values for fields marked with `configurable_fields()` or `configurable_alternatives()`.

### Configurable Fields

**`configurable_fields(**fields)`** (`repo://libs/core/langchain_core/runnables/base.py`)

Marks fields as runtime-configurable.

```python
model = ChatOpenAI(model="gpt-4").configurable_fields(
    model=ConfigurableField(
        id="model_name",
        name="Model Name",
        description="The model to use",
    )
)

result = model.invoke(
    "Hello",
    config={"configurable": {"model_name": "gpt-3.5-turbo"}}
)
```

**Behavior**:
- Fields exposed in `config_schema()`.
- Runtime values applied before invoke.

**`configurable_alternatives(**fields)`**

Alternative is similar but allows selecting from alternative implementations at runtime.

## Binding and Modification Methods

### bind

**`bind(**kwargs)`** (`repo://libs/core/langchain_core/runnables/base.py#L1903-L1936`)

Binds arguments to a Runnable, returning a new Runnable.

```python
model = ChatOpenAI().bind(stop=["three"])
```

**Use case**: Pass static arguments that aren't in chain output.

### with_config

**`with_config(config=None, **kwargs)`** (`repo://libs/core/langchain_core/runnables/base.py#L1937-L1961`)

Binds configuration to a Runnable.

```python
runnable = model.with_config(tags=["production"])
```

### with_types

**`with_types(*, input_type=None, output_type=None)`** (`repo://libs/core/langchain_core/runnables/base.py#L2131-L2152`)

Binds custom input and output type annotations.

### with_listeners

**`with_listeners(*, on_start=None, on_end=None, on_error=None)`** (`repo://libs/core/langchain_core/runnables/base.py#L1962-L2033`)

Binds lifecycle listeners.

```python
def on_start(run: Run):
    print(f"Started: {run.id}")

runnable = model.with_listeners(on_start=on_start)
```

**Listeners receive**: `Run` object with id, type, input, output, error, timing, tags, metadata.

### with_alisteners

**`with_alisteners(*, on_start=None, on_end=None, on_error=None)`** (`repo://libs/core/langchain_core/runnables/base.py#L2034-L2130`)

Async variant of `with_listeners`.

## Stream Transformations

### pick

**`pick(keys)`** (`repo://libs/core/langchain_core/runnables/base.py#L826-L900`)

Selects specific keys from dict output.

```python
chain | RunnablePick("key_a")
# Or: chain.pick(["key_a", "key_b"])
```

**Returns**: New Runnable outputting only specified keys.

### assign

**`assign(**kwargs)`** (`repo://libs/core/langchain_core/runnables/base.py#L900-L946`)

Adds new fields to dict output by invoking additional runnables.

```python
chain.assign(new_field=some_runnable)
```

**Behavior**: Output includes original fields plus computed new fields.

### map

**`map()`** (`repo://libs/core/langchain_core/runnables/base.py#L2220-L2242`)

Returns a Runnable that maps list inputs to list outputs.

```python
mapped = runnable.map()
mapped.invoke([1, 2, 3])  # [runnable.invoke(1), runnable.invoke(2), ...]
```

## Core Runnable Implementations

### RunnableLambda

**Location**: `repo://libs/core/langchain_core/runnables/base.py#L4707-L4850`

Wraps a Python callable into a Runnable.

```python
from langchain_core.runnables import RunnableLambda

runnable = RunnableLambda(lambda x: x + 1)
runnable.invoke(1)  # 2
```

**Features**:
- Detects async callables and provides native async support.
- If lambda returns a Runnable, that runnable is automatically invoked.
- Infers input/output types from signature for schema introspection.
- Separate `afunc` can be provided for async.

### RunnableGenerator

**Location**: `repo://libs/core/langchain_core/runnables/base.py`

Wraps generator functions (sync or async) for streaming.

```python
from langchain_core.runnables import RunnableGenerator

def token_stream(input: str) -> Iterator[str]:
    for token in input.split():
        yield token

runnable = RunnableGenerator(token_stream)
for chunk in runnable.stream("hello world"):
    print(chunk)
```

**Features**:
- Supports sync and async generators.
- Implements `transform` for end-to-end streaming.

### RunnablePassthrough

**Location**: `repo://libs/core/langchain_core/runnables/passthrough.py`

Passes input through unchanged.

```python
chain = prompt | {
    "original": RunnablePassthrough(),
    "result": model,
}
```

### RunnablePick

Selects and returns specific keys from dict input.

### RunnableAssign

Adds computed fields to dict output.

### RunnableBinding

Wraps a Runnable with bound arguments, config, or type overrides. Created by `bind()`, `with_config()`, `with_types()`, etc.

## Type Coercion: RunnableLike

**Location**: `repo://libs/core/langchain_core/runnables/base.py#L6612-L6668`

`RunnableLike` is a union type encompassing anything composable:

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
if isinstance(thing, Runnable):
    return thing
if is_async_generator(thing) or inspect.isgeneratorfunction(thing):
    return RunnableGenerator(thing)
if callable(thing):
    return RunnableLambda(thing)
if isinstance(thing, dict):
    return RunnableParallel(thing)
raise TypeError("Unsupported type")
```

This enables natural composition: `prompt | my_function | {"field": another_function}`.

## Design Invariants

### Immutability

Composition methods (`|`, `pipe`, `with_fallbacks`, `with_retry`, `pick`, `assign`) return new Runnables; they do not mutate originals. This enables safe reuse without side effects.

### Type Transparency

- `input_schema` and `output_schema` expose Pydantic models for validation and tooling.
- JSON Schema generation enables API documentation and OpenAPI integration.
- Type hints enable IDE autocompletion and static type checking.

### Lazy Execution

Composition with `|` creates structure but does not execute. Execution happens only on `invoke()`, `ainvoke()`, `batch()`, `abatch()`, `stream()`, or `astream()`.

### Streaming as First-Class

Streaming is not optional:
- Every Runnable exposes `stream()` and `astream()` for incremental output.
- The `transform()` method enables true progressive streaming through chains without blocking on full input.
- Default implementations ensure all runnables are composable in streaming contexts.

### Concurrency and Parallelism

- `batch()` uses `ThreadPoolExecutor` by default; subclasses override for backend optimization.
- `abatch()` uses `asyncio.gather()` with `max_concurrency` control.
- Parallel composition (`RunnableParallel`) executes all branches concurrently.
- Config threading enables fine-grained concurrency control.

### Uniformity

All Runnables implement the same interface:
- Sync and async variants for all operations.
- Single-input and batch operations share semantics.
- Streaming follows the same composition patterns as non-streaming.
- A single chain works in sync, async, batch, and streaming contexts without modification.

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
- `on_runnable_start`: Step begins.
- `on_runnable_end`: Step completes.
- `on_runnable_error`: Step fails.
- `on_llm_new_token`: Token streamed (LLM-specific).
- `on_chain_start`, `on_chain_end`, `on_tool_start`: Component-specific hooks.

**Behavior**:
- All composition operators automatically integrate callbacks.
- Enables end-to-end tracing of complex chains without modification.
- Intermediate steps and errors are automatically tracked.
- Custom callbacks enable user attribution, performance monitoring, and custom logging.

## Common Patterns

### Chaining with Type Safety

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel

class Summary(BaseModel):
    title: str
    key_points: list[str]

prompt = ChatPromptTemplate.from_template("Summarize: {text}")
model = ChatOpenAI(model="gpt-4")
parser = JsonOutputParser(pydantic_object=Summary)

chain = prompt | model | parser
result = chain.invoke({"text": "..."})  # Type: Summary
```

### Parallel Processing

```python
chain = input_processor | {
    "summary": summarizer,
    "entities": entity_extractor,
    "sentiment": sentiment_analyzer,
}
result = chain.invoke(data)  # {"summary": ..., "entities": ..., "sentiment": ...}
```

### Error Handling with Fallbacks

```python
chain = primary_model.with_fallbacks([fallback_1, fallback_2])
chain.invoke(input)  # Uses first successful model
```

### Conditional Routing

```python
router = RunnableBranch(
    (lambda x: "math" in x.lower(), math_chain),
    (lambda x: "text" in x.lower(), text_chain),
    fallback_chain,
)
result = router.invoke(user_query)
```

### Batch Processing

```python
results = chain.batch(inputs, config={"max_concurrency": 5})
# Process multiple inputs in parallel, limiting to 5 concurrent tasks
```

### Real-Time Streaming

```python
for chunk in chain.stream(input):
    print(chunk, end="", flush=True)
```

### Structured Event Streaming

```python
async for event in chain.astream_events(input, version="v2"):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="", flush=True)
    elif event["event"] == "on_chain_end":
        print("Chain complete:", event["data"]["output"])
```

## See Also

- [Runnables: Core Execution Abstraction](/openwiki/concepts/runnables.md) - Conceptual overview and design principles
- [LangChain Core API Docs](https://python.langchain.com/api_reference/core/runnables/) - Official API documentation
