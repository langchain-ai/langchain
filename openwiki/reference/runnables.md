---
type: "API Reference"
title: "Runnable: Complete API Reference"
description: "Complete API reference for the Runnable protocol, covering execution methods (invoke, batch, stream), async variants, composition operators, configuration, and lifecycle management."
tags: [runnable, lcel, api-reference, execution, protocol, composition]
verified:
  - by: openwiki/0.5.0
    at: 2026-11-21T08:14:55.070Z
sources:
  - id: openwiki-source-a1981e868973f6fd7f71e12e
    resource: repo://libs/core/langchain_core/runnables/base.py
generated: { by: "openwiki/0.5.0", at: "2026-11-21T08:14:55.070Z" }
---

# Runnable: Complete API Reference

The `Runnable` abstract base class (`repo://libs/core/langchain_core/runnables/base.py#L197-L330`) defines the foundational execution interface for all LangChain components. Every Runnable is generic over input and output types (`Runnable[Input, Output]`) and must implement only the abstract `invoke` method; all other execution methods have default implementations that subclasses can override for optimization.

This page documents all methods, configuration options, composition operators, and patterns essential to using and extending the Runnable protocol.

## Core Execution Methods

The Runnable protocol provides synchronous, asynchronous, batch, and streaming execution. All methods accept an optional `RunnableConfig` parameter that carries execution context (tags, metadata, callbacks, concurrency limits).

### Synchronous Single Invocation

**`invoke(input, config=None, **kwargs) → Output`** (`repo://libs/core/langchain_core/runnables/base.py#L949-L970`)

The abstract method that all subclasses must implement. Transforms a single input into an output synchronously.

```python
def invoke(
    self,
    input: Input,
    config: RunnableConfig | None = None,
    **kwargs: Any,
) -> Output:
    """Transform a single input into an output."""
```

**Parameters:**
- `input`: The data to transform.
- `config`: Optional `RunnableConfig` dictionary supporting:
  - `tags`: List of strings for tracing and observability.
  - `metadata`: Dictionary of metadata for observability and debugging.
  - `callbacks`: List of callback handlers for lifecycle events.
  - `max_concurrency`: Maximum parallel operations (for batch/stream methods).
  - `run_name`: Custom name for this execution run.
  - `run_id`: Explicitly set run identifier (overrides auto-generated ID).
  - `configurable`: Dictionary of configurable parameter overrides.
- `**kwargs`: Additional keyword arguments passed through to implementation.

**Returns:** Transformed output.

**Requirement:** Must be implemented by all `Runnable` subclasses. All other methods have default implementations that derive from `invoke`.

**Failure Semantics:** If `invoke` raises an exception, the caller receives it immediately.

### Asynchronous Single Invocation

**`ainvoke(input, config=None, **kwargs) → Coroutine[Output]`** (`repo://libs/core/langchain_core/runnables/base.py#L972-L993`)

Async variant of `invoke`. Default implementation delegates to `invoke` via `run_in_executor` (thread pool). Subclasses override for native async implementations.

```python
async def ainvoke(
    self,
    input: Input,
    config: RunnableConfig | None = None,
    **kwargs: Any,
) -> Output:
    """Transform a single input into an output asynchronously."""
```

**Parameters:** Same as `invoke`.

**Returns:** Coroutine yielding transformed output.

**Default Behavior:** Runs `invoke` in a thread pool executor, allowing non-blocking async contexts.

**Optimization:** Override for components with native async I/O (e.g., async HTTP clients, async database drivers).

### Batch Processing

**`batch(inputs, config=None, *, return_exceptions=False, **kwargs) → list[Output]`** (`repo://libs/core/langchain_core/runnables/base.py#L995-L1043`)

Processes multiple inputs in parallel. Default implementation calls `invoke` for each input using a thread pool executor.

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

**Parameters:**
- `inputs`: List of inputs to process.
- `config`: Single `RunnableConfig` (applied to all inputs) or list matching length of inputs. If None, uses default config.
- `return_exceptions`: If `True`, exceptions are returned in output list at corresponding indices. If `False` (default), raises on first exception.
- `**kwargs`: Additional arguments passed to each `invoke` call.

**Returns:** List of outputs, preserving input order.

**Default Behavior:** Uses `ThreadPoolExecutor` to invoke in parallel. Single input is processed without executor overhead. Empty input list returns empty output list.

**Optimization:** Override for components with batching-aware APIs (e.g., LLM batch endpoints, database bulk operations).

### Asynchronous Batch Processing

**`abatch(inputs, config=None, *, return_exceptions=False, **kwargs) → Coroutine[list[Output]]`** (`repo://libs/core/langchain_core/runnables/base.py#L1130-L1176`)

Async variant of `batch`. Default implementation calls `ainvoke` concurrently with `asyncio.gather`, respecting `max_concurrency` from config.

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

**Parameters:** Same as `batch`.

**Returns:** Coroutine yielding list of outputs.

**Concurrency Control:** `max_concurrency` from the first config (or first item if config is a list) limits parallelism. If `max_concurrency` is not set or is None, uses unlimited concurrency via `asyncio.gather`.

**Default Behavior:** Uses `asyncio.gather` with optional semaphore gating based on `max_concurrency`.

### Batch as Completed (Streaming Results)

**`batch_as_completed(inputs, config=None, *, return_exceptions=False, **kwargs) → Iterator[tuple[int, Output | Exception]]`** (`repo://libs/core/langchain_core/runnables/base.py#L1065-L1129`)

Synchronous batch that yields `(index, output)` tuples as they complete, allowing early processing without waiting for all inputs.

```python
def batch_as_completed(
    self,
    inputs: Sequence[Input],
    config: RunnableConfig | Sequence[RunnableConfig] | None = None,
    *,
    return_exceptions: bool = False,
    **kwargs: Any | None,
) -> Iterator[tuple[int, Output | Exception]]:
    """Run invoke in parallel, yielding results as they complete."""
```

**Parameters:** Same as `batch`.

**Yields:** Tuples of `(input_index, output)` in completion order, not input order. This allows processing results as soon as they are ready.

**Use Case:** Process results progressively without waiting for the slowest task; useful for display updates, real-time dashboards, or aggregating results incrementally.

### Asynchronous Batch as Completed

**`abatch_as_completed(inputs, config=None, *, return_exceptions=False, **kwargs) → AsyncIterator[tuple[int, Output | Exception]]`** (`repo://libs/core/langchain_core/runnables/base.py#L1198-L1256`)

Async variant of `batch_as_completed`. Yields results as coroutines complete, respecting concurrency limits.

**Parameters:** Same as `batch`.

**Yields:** Tuples of `(input_index, output)` in completion order. Uses `asyncio.as_completed` for iteration.

**Concurrency Control:** Respects `max_concurrency` from config via semaphore gating.

## Streaming Methods

Streaming methods yield outputs progressively as they are produced. Default implementations yield a single output; specialized components (LLMs, parsers) override to stream tokens or partial results.

### Synchronous Streaming

**`stream(input, config=None, **kwargs) → Iterator[Output]`** (`repo://libs/core/langchain_core/runnables/base.py#L1258-L1277`)

Yields partial outputs as they are produced. Default implementation yields a single output from `invoke`.

```python
def stream(
    self,
    input: Input,
    config: RunnableConfig | None = None,
    **kwargs: Any | None,
) -> Iterator[Output]:
    """Yield outputs as they are produced."""
```

**Returns:** Iterator yielding outputs. For LLMs, this yields tokens or chunks. For parsers, this yields partial parse results.

**Default Behavior:** Calls `invoke` once and yields the single result.

**Specialization:** LLMs and streaming parsers override to yield tokens or intermediate results progressively.

**Use Case:** Display streaming responses in real-time, process large outputs without buffering, or enable progressive results aggregation.

### Asynchronous Streaming

**`astream(input, config=None, **kwargs) → AsyncIterator[Output]`** (`repo://libs/core/langchain_core/runnables/base.py#L1279-L1298`)

Async variant of `stream`. Default implementation yields a single output from `ainvoke`.

```python
async def astream(
    self,
    input: Input,
    config: RunnableConfig | None = None,
    **kwargs: Any | None,
) -> AsyncIterator[Output]:
    """Yield outputs asynchronously as they are produced."""
```

**Returns:** Async iterator yielding outputs.

**Default Behavior:** Awaits `ainvoke` once and yields the single result.

### Transform: Iterator-to-Iterator Transformation

**`transform(input, config=None, **kwargs) → Iterator[Output]`** (`repo://libs/core/langchain_core/runnables/base.py#L1824-L1861`)

Maps an iterator of inputs to an iterator of outputs. Default buffers input and calls `stream`.

```python
def transform(
    self,
    input: Iterator[Input],
    config: RunnableConfig | None = None,
    **kwargs: Any | None,
) -> Iterator[Output]:
    """Transform inputs to outputs."""
```

**Parameters:**
- `input`: An iterator of inputs to the runnable.
- `config`: Configuration for the runnable.
- `**kwargs`: Additional keyword arguments.

**Returns:** Iterator of outputs.

**Default Behavior:** Accumulates all input using the `+` operator (via `_ChunkAccumulator`), then calls `stream` on the accumulated value. If input chunks are not addable (don't support `+`), retains only the last chunk.

**Specialization:** Override for components that can start producing output before all input is consumed. For example, a parser that processes streaming input could yield partial results as chunks arrive.

**Purpose:** Enables composing streaming chains where downstream runnables receive output chunks as upstream processing produces them, without buffering entire intermediate results.

### Async Transform

**`atransform(input, config=None, **kwargs) → AsyncIterator[Output]`** (`repo://libs/core/langchain_core/runnables/base.py#L1863-L1901`)

Async variant of `transform`. Maps async iterator of inputs to async iterator of outputs.

**Parameters:** Same as `transform`, with async iterators.

**Returns:** Async iterator of outputs.

**Default Behavior:** Accumulates all input, then calls `astream` on the accumulated value.

## Event Streaming (Advanced Observability)

### Async Stream Events

**`astream_events(input, config=None, *, version='v2', include_names=None, include_types=None, include_tags=None, exclude_names=None, exclude_types=None, exclude_tags=None, **kwargs) → AsyncIterator[StreamEvent]`** (`repo://libs/core/langchain_core/runnables/base.py#L1432-L1735`)

Streams structured lifecycle events from the runnable and its child runnables. Each event provides real-time visibility into execution progress, intermediate results, and errors.

```python
async def astream_events(
    self,
    input: Any,
    config: RunnableConfig | None = None,
    *,
    version: Literal["v1", "v2", "v3"] = "v2",
    include_names: Sequence[str] | None = None,
    include_types: Sequence[str] | None = None,
    include_tags: Sequence[str] | None = None,
    exclude_names: Sequence[str] | None = None,
    exclude_types: Sequence[str] | None = None,
    exclude_tags: Sequence[str] | None = None,
    **kwargs: Any,
) -> AsyncIterator[StreamEvent]:
    """Generate a stream of events."""
```

**Parameters:**
- `version`: Event schema version. `"v2"` (recommended); `"v1"` and `"v3"` supported for specific use cases.
- `include_names`: Only include events from runnables with matching names (exact match or substring).
- `include_types`: Only include events from runnables with matching types (e.g., "llm", "prompt", "parser").
- `include_tags`: Only include events from runnables with matching tags.
- `exclude_names`: Exclude events from runnables with matching names.
- `exclude_types`: Exclude events from runnables with matching types.
- `exclude_tags`: Exclude events from runnables with matching tags.

**Event Format:** Dictionary with keys:
- `event`: Event name (format: `on_[runnable_type]_(start|stream|end)`).
- `name`: Name of the runnable that generated the event.
- `run_id`: Randomly generated ID for this execution (unique per runnable instance).
- `parent_ids`: List of parent run IDs (v2 only). Root runnable has empty list. Order is root to immediate parent.
- `tags`: Tags of the runnable that generated the event.
- `metadata`: Metadata of the runnable.
- `data`: Event data (contents depend on event type). Contains `input`, `chunk`, or `output` keys.

**Use Case:** Real-time visibility into chain execution, intermediate results, error tracking, and performance monitoring. Enable dynamic streaming dashboards, debugging, and tracing.

### Sync Stream Events

**`stream_events(input, config=None, *, version='v2', ...) → Iterator[StreamEvent]`** (`repo://libs/core/langchain_core/runnables/base.py#L1763-L1822`)

Synchronous variant of `astream_events`. For `version='v3'`, subclasses that implement the v3 streaming protocol (e.g., `BaseChatModel`, `CompiledGraph`) override this method. Versions `'v1'` and `'v2'` are not supported on the sync path and raise `NotImplementedError`.

**Parameters:** Same as `astream_events`.

**Raises:** `NotImplementedError` for unsupported versions or when called on the base `Runnable` class (unless subclass implements v3).

## Schema Introspection

Runnables expose schema information for validation, tooling, and documentation.

### Input Schema

**`input_schema: TypeBaseModel`** (property, `repo://libs/core/langchain_core/runnables/base.py#L438-L441`)

Returns a Pydantic model for validating expected input shape.

**Use:** Validation, documentation, tool interfaces, schema export.

**`get_input_schema(config=None) → TypeBaseModel`** (`repo://libs/core/langchain_core/runnables/base.py#L443-L482`)

Get input schema for a specific config. Useful for runnables with configurable input schemas.

**Parameters:**
- `config`: Configuration that may affect input schema (for configurable runnables).

**Returns:** Pydantic model for input validation.

**`get_input_jsonschema(config=None) → dict`** (`repo://libs/core/langchain_core/runnables/base.py#L484-L512`)

Get JSON schema representation of input. Useful for API documentation, UI generation.

**Returns:** JSON schema dictionary (dict representation of Pydantic model schema).

### Output Schema

**`output_schema: TypeBaseModel`** (property, `repo://libs/core/langchain_core/runnables/base.py#L514-L520`)

Returns a Pydantic model for validating expected output shape.

**`get_output_schema(config=None) → TypeBaseModel`** (`repo://libs/core/langchain_core/runnables/base.py#L522-L561`)

Get output schema for a specific config.

**`get_output_jsonschema(config=None) → dict`** (`repo://libs/core/langchain_core/runnables/base.py#L563-L591`)

Get JSON schema representation of output.

### Type Properties

**`InputType: type`** (property, `repo://libs/core/langchain_core/runnables/base.py#L372-L405`)

The Python type annotation for accepted input. Extracted from generic parameters or function signatures.

**Raises:** `TypeError` if input type cannot be inferred. Override the property to specify manually.

**`OutputType: type`** (property, `repo://libs/core/langchain_core/runnables/base.py#L407-L436`)

The Python type annotation for produced output.

### Config Schema

**`config_schema(*, include=None) → type[BaseModel]`** (`repo://libs/core/langchain_core/runnables/base.py#L598-L639`)

Returns Pydantic model for configuration options accepted by this runnable.

**Parameters:**
- `include`: Optional list of configuration field names to include. If None, includes all.

**Returns:** Pydantic model class for config validation.

**`get_config_jsonschema(*, include=None) → dict`** (`repo://libs/core/langchain_core/runnables/base.py#L641-L655`)

Get JSON schema representation of config.

**`config_specs: list[ConfigurableFieldSpec]`** (property, `repo://libs/core/langchain_core/runnables/base.py#L593-L596`)

List of configurable field specs for runnables using `configurable_fields()` or `configurable_alternatives()`.

## Composition Operators and Methods

Runnables support operators and methods for declarative composition into chains.

### Pipe Operator

**`__or__(other) → RunnableSequence`** (`repo://libs/core/langchain_core/runnables/base.py#L712-L737`)

Chains runnables sequentially: `runnable1 | runnable2 | runnable3`. Output of one becomes input to next.

```python
def __or__(
    self,
    other: Runnable[Output, Other] | Callable[[Output], Other] | Mapping[str, Any],
) -> RunnableSerializable[Input, Other]:
    """Runnable pipe operator. Compose this Runnable with another."""
```

**Accepts:**
- `Runnable` instances: Chained sequentially.
- Callables (functions): Automatically wrapped in `RunnableLambda`.
- Dict literals: Automatically wrapped in `RunnableParallel`. Values are run with shared input.

**Returns:** `RunnableSequence` composing both runnables.

**Example:**
```python
chain = prompt | model | parser
result = chain.invoke({"topic": "AI"})
```

### Reverse Pipe Operator

**`__ror__(other) → RunnableSequence`** (`repo://libs/core/langchain_core/runnables/base.py#L761-L786`)

Right-associative pipe: `x | runnable` where `x` is not a `Runnable`. Enables piping from non-Runnable objects.

**Example:**
```python
chain = {"text": "hello"} | my_runnable  # dict becomes RunnableParallel
```

### Pipe Method

**`pipe(*others, name=None) → RunnableSequence`** (`repo://libs/core/langchain_core/runnables/base.py#L788-L827`)

Explicit pipe method: `runnable.pipe(other1, other2, ...)` equivalent to `runnable | other1 | other2 | ...`.

**Parameters:**
- `*others`: Other `Runnable` or `Runnable`-like objects to compose.
- `name`: Optional name for the resulting `RunnableSequence`.

**Returns:** New `RunnableSequence`.

**Example:**
```python
sequence = runnable_1.pipe(runnable_2, runnable_3)
# Equivalent to: sequence = runnable_1 | runnable_2 | runnable_3
```

## Configuration and Binding

Methods for modifying runnable behavior without creating new subclasses.

### Bind Arguments

**`bind(**kwargs) → Runnable[Input, Output]`** (`repo://libs/core/langchain_core/runnables/base.py#L1903-L1935`)

Binds keyword arguments to a runnable, returning a new runnable that invokes with those args bound.

```python
def bind(self, **kwargs: Any) -> Runnable[Input, Output]:
    """Bind arguments to a Runnable."""
```

**Use Case:** Pass parameters not in chain input (e.g., `model.bind(temperature=0.5, stop=["\\n"])`). Common for LLM parameters, tool options, or parser config.

**Returns:** New `RunnableBinding` wrapping this runnable with bound kwargs.

**Example:**
```python
model = ChatOpenAI(model="gpt-4o")
bound_model = model.bind(temperature=0, top_p=0.5)
result = (prompt | bound_model | parser).invoke({"query": "..."})
```

### Bind Config

**`with_config(config=None, **kwargs) → Runnable[Input, Output]`** (`repo://libs/core/langchain_core/runnables/base.py#L1937-L1960`)

Binds configuration (tags, metadata, callbacks, max_concurrency) to a runnable.

```python
def with_config(
    self,
    config: RunnableConfig | None = None,
    **kwargs: Any,
) -> Runnable[Input, Output]:
    """Bind config to a Runnable."""
```

**Use Case:** Add tracing tags, metadata, or callbacks to a chain segment. Control concurrency limits for specific components.

**Returns:** New `RunnableBinding` with bound config.

**Example:**
```python
traced_chain = chain.with_config(
    tags=["chain-v1"],
    metadata={"environment": "production"},
    max_concurrency=5
)
```

## Lifecycle and Error Handling

Methods for adding lifecycle hooks and resilience patterns.

### Listeners (Sync)

**`with_listeners(*, on_start=None, on_end=None, on_error=None) → Runnable[Input, Output]`** (`repo://libs/core/langchain_core/runnables/base.py#L1962-L2032`)

Binds lifecycle callbacks (start, end, error) to a runnable for sync execution.

```python
def with_listeners(
    self,
    *,
    on_start: Callable[[Run], None] | Callable[[Run, RunnableConfig], None] | None = None,
    on_end: Callable[[Run], None] | Callable[[Run, RunnableConfig], None] | None = None,
    on_error: Callable[[Run], None] | Callable[[Run, RunnableConfig], None] | None = None,
) -> Runnable[Input, Output]:
    """Bind lifecycle listeners to a Runnable."""
```

**Parameters:**
- `on_start`: Called before execution with `Run` object. Receives `(Run)` or `(Run, RunnableConfig)`.
- `on_end`: Called after successful execution with `Run` object.
- `on_error`: Called on exception with `Run` object (which contains the error).

**Run Object:** Contains `id`, `type`, `input`, `output`, `error`, `start_time`, `end_time`, `tags`, `metadata`.

**Returns:** New `RunnableBinding` with listener callbacks.

**Example:**
```python
def log_start(run: Run) -> None:
    print(f"Starting run {run.id}")

def log_error(run: Run) -> None:
    print(f"Error in {run.id}: {run.error}")

chain = my_runnable.with_listeners(on_start=log_start, on_error=log_error)
```

### Async Listeners

**`with_alisteners(*, on_start=None, on_end=None, on_error=None) → Runnable[Input, Output]`** (`repo://libs/core/langchain_core/runnables/base.py#L2034-L2135`)

Async variant of `with_listeners` for use in async execution contexts. Callbacks are async callables.

**Parameters:** Same as `with_listeners`, with async callables.

### Retry Policy

**`with_retry(*, stop_after_attempt=3, wait_exponential_jitter=True, retry_if_exception_type=(Exception,), exponential_jitter_params=None) → Runnable[Input, Output]`** (`repo://libs/core/langchain_core/runnables/base.py#L2153-L2218`)

Binds automatic retry logic to a runnable with exponential backoff.

```python
def with_retry(
    self,
    *,
    stop_after_attempt: int = 3,
    wait_exponential_jitter: bool = True,
    retry_if_exception_type: tuple[type[BaseException], ...] = (Exception,),
    exponential_jitter_params: ExponentialJitterParams | None = None,
) -> Runnable[Input, Output]:
    """Bind a retry policy to the underlying Runnable."""
```

**Parameters:**
- `stop_after_attempt`: Max retry attempts (default 3). Includes initial attempt.
- `wait_exponential_jitter`: Use exponential backoff with jitter (default True).
- `retry_if_exception_type`: Tuple of exception types to retry on (default catches all `Exception`). Can be a callable `(Exception) -> bool`.
- `exponential_jitter_params`: Custom backoff parameters (initial, max, exp_base, jitter).

**Returns:** New `RunnableRetry` wrapper.

**Example:**
```python
resilient_chain = unreliable_runnable.with_retry(
    stop_after_attempt=5,
    retry_if_exception_type=(TimeoutError, ConnectionError)
)
```

### Fallbacks

**`with_fallbacks(fallbacks, *, exceptions_to_handle=(Exception,), exception_key=None) → RunnableWithFallbacks[Input, Output]`** (`repo://libs/core/langchain_core/runnables/base.py#L2243-L2319`)

Chains fallback runnables to try on failure.

```python
def with_fallbacks(
    self,
    fallbacks: Sequence[Runnable[Input, Output]],
    *,
    exceptions_to_handle: tuple[type[BaseException], ...] = (Exception,),
    exception_key: str | None = None,
) -> RunnableWithFallbacksT[Input, Output]:
    """Add fallbacks to a Runnable."""
```

**Parameters:**
- `fallbacks`: Sequence of runnables to try in order on failure.
- `exceptions_to_handle`: Exception types to catch and trigger fallback (default all `Exception`).
- `exception_key`: If set (string), exception is passed to fallbacks under this key in input dict (input must be dict).

**Use Case:** Graceful degradation. Try primary method, fall back to alternative on failure (e.g., try GPT-4, fall back to GPT-3.5-turbo).

**Returns:** New `RunnableWithFallbacks` wrapper.

**Example:**
```python
chain = primary_model.with_fallbacks(
    [fallback_model_1, fallback_model_2],
    exceptions_to_handle=(RateLimitError, APIError)
)
```

### Map

**`map() → Runnable[Sequence[Input], list[Output]]`** (`repo://libs/core/langchain_core/runnables/base.py#L2220-L2241`)

Returns a runnable that applies this runnable to each element of an input sequence.

```python
def map(self) -> Runnable[Sequence[Input], list[Output]]:
    """Return a new Runnable that maps a list of inputs to list of outputs."""
```

**Equivalent to:** Calling `invoke` on each element and collecting results into a list.

**Returns:** New `RunnableEach` wrapper.

**Example:**
```python
single_processor = RunnableLambda(lambda x: x + 1)
batch_processor = single_processor.map()
result = batch_processor.invoke([1, 2, 3])  # [2, 3, 4]
```

## Utility Methods

### Get Name

**`get_name(suffix=None, *, name=None) → str`** (`repo://libs/core/langchain_core/runnables/base.py#L334-L370`)

Retrieve the runnable's name, used for tracing and debugging.

**Parameters:**
- `suffix`: Optional suffix appended to name (title-cased for uppercase names, underscore-lowercased for lowercase).
- `name`: Override with custom name.

**Returns:** String name of runnable.

**Default Behavior:** Returns the runnable's `name` attribute if set, otherwise the class name. Useful for debugging, logging, and tracing.

### Get Graph

**`get_graph(config=None) → Graph`** (`repo://libs/core/langchain_core/runnables/base.py#L657-L676`)

Returns a graph representation of the runnable's structure.

**Use Case:** Visualization, debugging complex compositions. Useful for rendering chain structure diagrams.

### Get Prompts

**`get_prompts(config=None) → list[BasePromptTemplate]`** (`repo://libs/core/langchain_core/runnables/base.py#L678-L689`)

Extracts all prompt templates used by this runnable and its children.

**Returns:** List of `BasePromptTemplate` instances.

**Use Case:** Dynamic prompt management, testing prompt variations.

## Helper Methods for Subclasses

Subclasses override `invoke` and typically use these helpers to manage config, callbacks, and error handling.

**`_call_with_config(func, input_, config, run_type=None, serialized=None, **kwargs) → Output`** (`repo://libs/core/langchain_core/runnables/base.py#L2323-L2370`)

Wraps a callable with config management, callbacks, and error handling. Use in `invoke` implementations.

**Purpose:** Automatically manages callback lifecycle (on_chain_start, on_chain_end, on_chain_error), config context, and error propagation.

**`_acall_with_config(func, input_, config, run_type=None, serialized=None, **kwargs) → Coroutine[Output]`** (`repo://libs/core/langchain_core/runnables/base.py#L2372-L2413`)

Async variant for `ainvoke` implementations.

**`_batch_with_config(func, inputs, config, *, return_exceptions=False, run_type=None, **kwargs) → list[Output]`** (`repo://libs/core/langchain_core/runnables/base.py#L2415-L2480`)

Wraps batch processing function with callback management. Use in `batch` implementations.

**`_abatch_with_config(func, inputs, config, *, return_exceptions=False, run_type=None, **kwargs) → Coroutine[list[Output]]`** (`repo://libs/core/langchain_core/runnables/base.py#L2482-L2555`)

Async variant for `abatch` implementations.

**`_transform_stream_with_config(inputs, transformer, config, run_type=None, **kwargs) → Iterator[Output]`** (`repo://libs/core/langchain_core/runnables/base.py#L2557-L2630`)

Wraps streaming transformer function with callback lifecycle. Use in `stream` and `transform` implementations.

## Implementation Patterns

### Minimal Runnable

Override only `invoke`:

```python
from langchain_core.runnables import Runnable, RunnableConfig
from typing import Any

class MyRunnable(Runnable[str, int]):
    def invoke(
        self,
        input: str,
        config: RunnableConfig | None = None,
        **kwargs: Any
    ) -> int:
        return len(input)
```

Async, batch, stream, and other methods work automatically with defaults.

### Optimized Runnable

Override methods for efficiency:

```python
class MyRunnable(Runnable[str, int]):
    def invoke(self, input: str, config: RunnableConfig | None = None, **kwargs: Any) -> int:
        return len(input)
    
    async def ainvoke(self, input: str, config: RunnableConfig | None = None, **kwargs: Any) -> int:
        # Native async implementation (not thread pool)
        return len(input)
    
    def batch(self, inputs: list[str], config: RunnableConfig | list[RunnableConfig] | None = None, *, return_exceptions: bool = False, **kwargs: Any) -> list[int]:
        # Batch optimization
        if return_exceptions:
            return [len(inp) for inp in inputs]
        else:
            return [len(inp) for inp in inputs]
    
    def stream(self, input: str, config: RunnableConfig | None = None, **kwargs: Any):
        # Streaming: yield chunks
        for i, char in enumerate(input):
            yield i  # Example: yield index as we process
```

## Composition Primitives (LCEL)

The `Runnable` protocol supports composition primitives:

- **Sequential (RunnableSequence):** Chain with `|` operator. Output of each becomes input to next.
- **Parallel (RunnableParallel):** Use dict literal `{ "key1": runnable1, "key2": runnable2 }` to invoke runnables concurrently with shared input.
- **Branching:** Conditional routing using mapping or custom logic (see `RunnableBranch`).
- **Fallback:** Try primary, fall back to secondary on failure (see `with_fallbacks`).
- **Retry:** Automatic exponential backoff retry (see `with_retry`).

All composed chains inherit full sync, async, batch, and streaming support automatically from the protocol.

## Configuration Threading

Runnables thread configuration through compositions. `RunnableConfig` parameters carry:

- **Tracing:** `tags`, `metadata` for logging and observability.
- **Concurrency:** `max_concurrency` limits parallel operations in batch/stream.
- **Callbacks:** Lifecycle listeners across the chain.
- **Configurable parameters:** Dynamic, per-runnable overrides.

Configuration is passed from parent to child runnables, allowing uniform control across complex chains.

## Relationship to Related Pages

- **[Runnables: Core Execution Abstraction](/openwiki/concepts/runnables.md):** Conceptual overview of Runnables and LCEL.
- **[Composability: Chaining & Piping Runnables](/openwiki/workflows/composability.md):** Practical composition patterns, composition operators, and design examples.
