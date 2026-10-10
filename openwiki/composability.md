---
type: "Concept"
title: "Composability: Chains & Workflows"
description: "Building applications by composing runnables: pipe operator, mapping, branching, error handling, and common patterns."
tags: ["composability", "LCEL", "runnables", "chaining", "operators", "branching", "error-handling"]
sources:
  - id: openwiki-source-bade918c6ca6cdcb4bfd3ed4
    resource: repo://libs/core/langchain_core/load/dump.py
  - id: openwiki-source-2b02620f51e06ab62ea15911
    resource: repo://libs/core/langchain_core/load/load.py
  - id: openwiki-source-a1981e868973f6fd7f71e12e
    resource: repo://libs/core/langchain_core/runnables/base.py
  - id: openwiki-source-48e94bbe49ab4f33ba87e9cb
    resource: repo://libs/core/langchain_core/runnables/branch.py
  - id: openwiki-source-f9f4c1dc4f9cdf80d824ce15
    resource: repo://libs/core/langchain_core/runnables/fallbacks.py
  - id: openwiki-source-ebe3f825462d0b4a14ee3717
    resource: repo://libs/core/langchain_core/runnables/retry.py
  - id: openwiki-source-de6c904bd0171642bd50f6d9
    resource: repo://libs/core/langchain_core/runnables/router.py
generated: { by: "openwiki/0.5.0", at: "2026-10-10T08:25:28.570Z" }
verified:
  - by: openwiki/0.5.0
    at: 2026-10-10T08:25:28.570Z
---


## Overview

Composability is the core feature of LangChain's Runnable protocol: the ability to declaratively chain, parallelize, and conditionally route components. Every composed chain automatically inherits sync (`invoke`), async (`ainvoke`), batch (`batch`/`abatch`), and streaming (`stream`/`astream`) capabilities—with optimizations for efficiency.

The two main composition primitives are **`RunnableSequence`** (sequential chaining via the `|` operator) and **`RunnableParallel`** (parallel execution via dict syntax). Conditional routing is achieved with **`RunnableBranch`** and **`RouterRunnable`**.

## Sequential Composition: The `|` Operator

The **pipe operator** (`|`) chains Runnables in sequence, with each step's output becoming the next step's input. This is the most common composition pattern.

```python
from langchain_core.runnables import RunnableLambda

add_one = RunnableLambda(lambda x: x + 1)
mul_two = RunnableLambda(lambda x: x * 2)

sequence = add_one | mul_two
sequence.invoke(1)  # (1 + 1) * 2 = 4
```

The `|` operator creates a **`RunnableSequence`**, which:
- Invokes each step in order, passing output to the next input
- Flattens nested sequences for efficiency
- Automatically preserves streaming properties if all steps support the `transform` method
- Supports both sync and async execution

### Data Flow

```
Input → Step 1 → Step 2 → Step 3 → Output
```

When a dict is piped into a sequence, it becomes a **`RunnableParallel`**:

```python
sequence = add_one | {
    "mul_2": RunnableLambda(lambda x: x * 2),
    "mul_5": RunnableLambda(lambda x: x * 5),
}
sequence.invoke(1)  # {'mul_2': 4, 'mul_5': 10}
```

## Parallel Composition: Branching with `+` and Dict Syntax

Parallel execution invokes multiple Runnables concurrently on the **same input**. This is achieved via dict literals within a sequence or directly with **`RunnableParallel`**.

### Dict Literal Syntax

```python
from langchain_core.runnables import RunnableLambda, RunnableParallel

add_one = RunnableLambda(lambda x: x + 1)
mul_two = RunnableLambda(lambda x: x * 2)
mul_three = RunnableLambda(lambda x: x * 3)

# Dict syntax creates a RunnableParallel
sequence = add_one | {
    "mul_2": mul_two,
    "mul_3": mul_three,
}
sequence.invoke(1)
# Output: {'mul_2': 4, 'mul_3': 6}
```

### Explicit RunnableParallel

```python
parallel = RunnableParallel(
    mul_2=mul_two,
    mul_3=mul_three,
)
parallel.invoke(2)
# Output: {'mul_2': 4, 'mul_3': 6}
```

### Concurrent Execution

- **`RunnableParallel`** creates independent input copies for each branch using `atee` (async) or `safetee` (sync)
- Each branch executes concurrently, with chunks yielded in the order they complete
- For async streaming, tasks are managed with `asyncio.wait(return_when=FIRST_COMPLETED)` to emit output as soon as any branch produces a chunk
- The final result is a dict combining outputs from all branches

## Mapping: `.map()` for Sequential Processing

The **`.map()`** method wraps a Runnable to process a list of inputs sequentially:

```python
from langchain_core.runnables import RunnableLambda

add_one = RunnableLambda(lambda x: x + 1)

# Create a mapped version that processes lists
mapper = add_one.map()
mapper.invoke([1, 2, 3])  # [2, 3, 4]
```

Unlike **`batch()`** which processes multiple inputs in parallel, **`.map()`**:
- Processes inputs sequentially (one at a time via `invoke`)
- Returns a list of outputs in the same order as inputs
- Useful when you want to maintain order without parallelism
- Can be used in chains: `sequence.map().invoke(inputs)`

## Batching: Parallel Invocation over Multiple Inputs

Batching processes multiple inputs efficiently through a pipeline. Unlike parallel branching, batching applies the **same sequence** to each input in parallel.

### Sync Batch

```python
sequence = add_one | mul_two
results = sequence.batch([1, 2, 3])
# [4, 6, 8]  # Each input processed in parallel via thread pool
```

### Async Batch

```python
results = await sequence.abatch([1, 2, 3])
# [4, 6, 8]
```

### Implementation

- Default `batch` uses a thread pool executor via `get_executor_for_config`
- `abatch` uses `asyncio.gather` with concurrency control via `max_concurrency`
- Each step in the sequence batches its inputs independently
- **`RunnableSequence`** calls `batch` on each step in order, feeding outputs to the next

## Streaming: Token-by-Token Output

Streaming emits output chunks as they are produced, enabling real-time responses from LLMs and other sequential generators.

### Stream Method

```python
for chunk in sequence.stream(1):
    print(chunk)  # Intermediate outputs as they become available
```

### Astream Method (Async)

```python
async for chunk in sequence.astream(1):
    print(chunk)  # Non-blocking iteration
```

### Streaming Pipeline

A **`RunnableSequence`** preserves streaming properties:
- If all steps implement `transform` (which processes `Iterator[Input] → Iterator[Output]`), streaming passes through the entire pipeline
- If any step doesn't support `transform`, streaming blocks until that step completes, then resumes
- **`RunnableLambda`** does not implement `transform` by default; use **`RunnableGenerator`** for custom streaming logic

### Example: Prompt → Model → Parser

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("What is {topic}?")
model = ChatOpenAI()
parser = StrOutputParser()

chain = prompt | model | parser

# Stream tokens as the model generates them
for chunk in chain.stream({"topic": "composability"}):
    print(chunk, end="", flush=True)
```

In this chain:
1. `ChatPromptTemplate` formats the input dict into a string prompt
2. `ChatOpenAI` streams tokens as they arrive from the API
3. `StrOutputParser` passes tokens through unchanged

Tokens flow end-to-end without waiting for the full response.

## Conditional Routing: RunnableBranch and RouterRunnable

Conditional logic routes inputs to different branches based on predicates.

### RunnableBranch: Predicate-Based Routing

A **`RunnableBranch`** evaluates a series of conditions in order and executes the first matching branch:

```python
from langchain_core.runnables import RunnableBranch, RunnableLambda

branch = RunnableBranch(
    (lambda x: isinstance(x, int), RunnableLambda(lambda x: x * 2)),
    (lambda x: isinstance(x, str), RunnableLambda(lambda x: x.upper())),
    RunnableLambda(lambda x: "unknown"),  # Default branch
)

branch.invoke(5)        # 10
branch.invoke("hello")  # "HELLO"
branch.invoke(None)     # "unknown"
```

**Key characteristics**:
- Conditions are evaluated sequentially
- The first truthy condition selects its corresponding Runnable
- Conditions can be plain functions or Runnables that return `bool`
- If no condition matches, the default branch (last argument) executes
- Both condition evaluation and branch execution support sync and async

### RouterRunnable: Key-Based Routing

A **`RouterRunnable`** routes to one of several branches based on a string key extracted from the input:

```python
from langchain_core.runnables.router import RouterRunnable
from langchain_core.runnables import RunnableLambda

add = RunnableLambda(lambda x: x + 1)
square = RunnableLambda(lambda x: x ** 2)

router = RouterRunnable(runnables={"add": add, "square": square})
router.invoke({"key": "square", "input": 3})  # 9
router.invoke({"key": "add", "input": 3})     # 4
```

**Key characteristics**:
- Input must be a dict with two fields:
  - `"key"`: The string routing key (selects which Runnable to invoke)
  - `"input"`: The data to pass to the selected Runnable
- The `runnables` dict maps routing keys to Runnable implementations
- If the key is not found in `runnables`, an error is raised
- Useful for choosing between implementations at runtime based on runtime data

## Composition with RunnablePassthrough

**`RunnablePassthrough`** forwards inputs unchanged or with additional keys, useful for preserving context in parallel branches:

```python
from langchain_core.runnables import RunnablePassthrough

chain = (
    RunnableLambda(lambda x: x + 1)
    | {
        "original": RunnablePassthrough(),
        "modified": RunnableLambda(lambda x: x * 2),
    }
)

chain.invoke(5)
# {'original': 6, 'modified': 12}
```

Here, the passthrough preserves the intermediate result for reuse by another branch.

## Async Equivalents

Every method has an async counterpart:

| Sync | Async |
|------|-------|
| `invoke(input)` | `ainvoke(input)` |
| `batch(inputs)` | `abatch(inputs)` |
| `stream(input)` | `astream(input)` |
| `transform(Iterator[Input])` | `atransform(AsyncIterator[Input])` |

Async methods integrate with the callback system and execute concurrency-aware batching via `asyncio.gather`.

## Variable Binding and Context Flow

In composed chains, data flows through steps along with execution context. Each step receives the output of the previous step as its input.

### Context Propagation

When a chain invokes, **`RunnableSequence`** creates a callback hierarchy for tracing:
- Each step is marked as a child run using `run_manager.get_child(f"seq:step:{i + 1}")`
- Callbacks, tags, and metadata flow through the chain via `RunnableConfig`
- `patch_config` updates the config for each step while preserving parent context

```python
from langchain_core.runnables import RunnableLambda

# Context flows through each step
step1 = RunnableLambda(lambda x: x + 1)
step2 = RunnableLambda(lambda x: x * 2)
chain = step1 | step2

# Invoke with tracing config
result = chain.invoke(
    5, 
    config={
        "run_name": "my_chain",
        "callbacks": [my_tracer],
        "tags": ["prod"],
    }
)
# Each step runs with inherited config while reporting to callbacks
```

### Dict Composition and Key Selection

When using dict syntax in a sequence, each dict key becomes a separate branch context:

```python
chain = step1 | {
    "result_a": step2,
    "result_b": step3,
}

# Output combines results from both branches
output = chain.invoke(input)  # {'result_a': ..., 'result_b': ...}
```

Each branch (`result_a`, `result_b`) appears as a separate child run in the callback trace.

## Fallback Patterns

Fallbacks provide resilience by switching to alternative Runnables when one fails, without retrying the same component.

### RunnableWithFallbacks Mechanics

**`RunnableWithFallbacks`** wraps a primary Runnable and a list of fallbacks:

1. Tries the primary Runnable first
2. On an exception matching `exceptions_to_handle`, attempts each fallback in order
3. Stops on the first successful execution
4. If all fail, raises the first exception encountered

Key attributes:
- **`runnable`**: The primary Runnable to execute
- **`fallbacks`**: A sequence of fallback Runnables (ordered)
- **`exceptions_to_handle`**: Tuple of exception types to trigger fallback (default: all exceptions)
- **`exception_key`**: Optional string key to pass the caught exception to fallbacks in the input dict

### Fallback at Component Level

```python
from langchain_core.runnables import RunnableLambda

primary_llm = ChatOpenAI(model="gpt-4")
fallback_llm = ChatAnthropic(model="claude-3-sonnet")

resilient_llm = primary_llm.with_fallbacks(
    [fallback_llm],
    exceptions_to_handle=(APIConnectionError,),
)

output = resilient_llm.invoke("What is composability?")
# Uses primary_llm; falls back to fallback_llm if APIConnectionError occurs
```

### Multiple Fallbacks with Provider Failover

Fallbacks are tried in order until one succeeds:

```python
model = ChatOpenAI().with_fallbacks([
    ChatAnthropic(),        # Try second
    ChatClaude(),          # Try third
    ChatCohere(),          # Try fourth
    RunnableLambda(default_response),  # Final fallback
])
```

The chain tries each fallback sequentially until one returns successfully or all are exhausted.

### Fallback at Chain Level

```python
# Construct a chain with fallback
chain_with_fallback = (
    prompt 
    | resilient_llm 
    | parser
).with_fallbacks([
    RunnableLambda(lambda x: "Service unavailable")
])

output = chain_with_fallback.invoke({"topic": "composability"})
# If the entire chain fails, returns fallback response
```

### Exception Passing to Fallbacks

Use `exception_key` to pass the caught exception to fallbacks as part of the input:

```python
def fallback_with_context(input_dict):
    error = input_dict.get("error")
    if isinstance(error, APIConnectionError):
        return "API is currently unavailable. Please try again later."
    return "An unexpected error occurred."

chain = (
    prompt | llm | parser
).with_fallbacks(
    [RunnableLambda(fallback_with_context)],
    exception_key="error",  # Pass exception under "error" key
)

# Input becomes {"topic": "...", "error": <caught exception>}
```

This requires all Runnables (primary and fallbacks) to accept a dictionary as input.

## Retry Patterns

Retry logic automatically re-invokes a Runnable on failure, with configurable backoff and exception filtering.

### Retry at Component Level

```python
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.retry import ExponentialJitterParams

llm = ChatOpenAI(model="gpt-4")

resilient_llm = llm.with_retry(
    retry_if_exception_type=(APIConnectionError, TimeoutError),
    max_attempt_number=3,
    wait_exponential_jitter=True,
    exponential_jitter_params={"initial": 1, "max": 10},
)

output = resilient_llm.invoke("What is composability?")
# Retries up to 3 times on transient errors with exponential backoff
```

### RunnableRetry Implementation

**`RunnableRetry`** wraps any Runnable and applies retry logic using `tenacity`:

- **`retry_exception_types`**: Tuple of exception types to retry on (default: all exceptions)
- **`max_attempt_number`**: Maximum retry attempts (default: 3)
- **`wait_exponential_jitter`**: Enable exponential backoff with jitter (default: True)
- **`exponential_jitter_params`**: Customize backoff parameters (`initial`, `max`, `exp_base`, `jitter`)

Retries are tracked in callbacks with tags like `retry:attempt:2` for visibility in traces.

### Retry Strategy: Transient vs. Fatal Errors

Best practice: retry only on transient errors, not fatal ones:

```python
model = ChatOpenAI().with_retry(
    retry_if_exception_type=(
        APIConnectionError,    # Transient: network issues
        TimeoutError,          # Transient: slow API
        RateLimitError,        # Transient: quota exceeded
    ),
    max_attempt_number=5,
    exponential_jitter_params={"initial": 0.5, "max": 30},
)

# Do NOT retry on invalid input or authentication errors—these are fatal
```

### Retry at Chain Level

```python
chain = (
    prompt 
    | llm.with_retry(max_attempt_number=3)  # Retry at LLM level
    | parser
)

# Better than retrying the whole chain, which wastes time on non-failing steps
```

## Combining Retry and Fallback

For maximum resilience, combine retry and fallback strategies:

```python
from langchain_openai import ChatOpenAI, ChatAnthropic

# Retry transient failures on each provider
primary_llm = ChatOpenAI().with_retry(
    retry_if_exception_type=(APIConnectionError, TimeoutError),
    max_attempt_number=3,
)

fallback_llm = ChatAnthropic().with_retry(
    retry_if_exception_type=(APIConnectionError, TimeoutError),
    max_attempt_number=3,
)

# Fall back if primary provider is unavailable
resilient_model = primary_llm.with_fallbacks([fallback_llm])

chain = prompt | resilient_model | parser
```

This approach:
1. Retries transient failures on the primary LLM (network glitches, temporary outages)
2. Falls back to an alternative provider if the primary is persistently down
3. Ensures reasonable wait times via exponential backoff
4. Provides observability through callback tags (`retry:attempt:N`)

## Chaining Patterns

### Common Pattern: Prompt → Model → Parser

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

chain = (
    ChatPromptTemplate.from_template("What is {topic}?")
    | ChatOpenAI()
    | StrOutputParser()
)

# Single invoke
output = chain.invoke({"topic": "LLMs"})

# Batch process
outputs = chain.batch([{"topic": "LLMs"}, {"topic": "Vectors"}])

# Stream tokens
for chunk in chain.stream({"topic": "LLMs"}):
    print(chunk, end="", flush=True)
```

### Fan-Out / Fan-In: Parallel Processing

```python
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

chain = (
    RunnablePassthrough()
    | {
        "summary": RunnableLambda(summarize),
        "entities": RunnableLambda(extract_entities),
        "sentiment": RunnableLambda(analyze_sentiment),
    }
)

result = chain.invoke(text)
# {'summary': '...', 'entities': [...], 'sentiment': 'positive'}
```

### Conditional Execution

```python
from langchain_core.runnables import RunnableBranch

route_logic = RunnableBranch(
    (lambda x: "math" in x.lower(), math_chain),
    (lambda x: "code" in x.lower(), code_chain),
    general_chain,
)

output = route_logic.invoke("How do I calculate factorial?")
```

## Type Safety and Schema Inference

Chains infer input and output types from their components:

```python
sequence = add_one | mul_two

# Access inferred schemas
print(sequence.input_schema)   # Pydantic model for input
print(sequence.output_schema)  # Pydantic model for output
print(sequence.input_schema.model_json_schema())
```

This enables validation and documentation without explicit type annotations.

## Optimization and Flattening

**`RunnableSequence`** automatically flattens nested sequences:

```python
# These are equivalent:
chain1 = step1 | step2 | step3
chain2 = step1 | (step2 | step3)
chain3 = (step1 | step2) | step3
```

All produce a single flat sequence with steps `[step1, step2, step3]`, avoiding unnecessary nesting overhead.

## Serialization and Deserialization

Composed chains are serializable via the LangChain serialization system for persistence and transport:

### Serialization: `dumpd` and `dumps`

The `dumpd` and `dumps` functions serialize Runnable chains to dict and JSON string representations:

```python
from langchain_core.load import dumpd, dumps
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

chain = (
    ChatPromptTemplate.from_template("What is {topic}?")
    | ChatOpenAI()
    | StrOutputParser()
)

# Serialize to dict
chain_dict = dumpd(chain)

# Serialize to JSON string
chain_json = dumps(chain, pretty=True)
```

The serialized form preserves:
- Component types and class identifiers (via `lc_id` field in the serialized output)
- Constructor arguments and configuration
- Nested composition structure

### Deserialization: `load`

The `load` and `loads` functions reconstruct Runnable chains from serialized data:

```python
from langchain_core.load import load, loads

# Load from dict
chain = load(chain_dict)

# Load from JSON string
chain = loads(chain_json)

# Control which classes can be deserialized (for security):
# - 'messages': only message classes (safe for untrusted input)
# - 'core': core LangChain classes (default, unsafe with untrusted data)
# - 'all': all registered classes (unsafe with untrusted data)
chain = loads(chain_json, allowed_objects='core')

# Or provide an explicit list of allowed classes
chain = loads(chain_json, allowed_objects=[ChatOpenAI, StrOutputParser])
```

**Security Note**: Only deserialize JSON from trusted sources. If the source is untrusted, restrict `allowed_objects` to `'messages'` or an explicit list of classes without dangerous constructor arguments.

## Debugging

Enable debug output and tracing for chains:

```python
from langchain_core.globals import set_debug

set_debug(True)  # Print intermediate results
chain.invoke(input)

# Or use callbacks:
from langchain_core.tracers import ConsoleCallbackHandler

chain.invoke(input, config={"callbacks": [ConsoleCallbackHandler()]})
```

Use `get_graph()` to visualize chain structure via LangSmith or other tracing tools.

## Extension: Custom Runnables

Implement **`Runnable`** to create custom components:

```python
from langchain_core.runnables import Runnable, RunnableConfig
from typing import Iterator

class CustomRunnable(Runnable[str, int]):
    def invoke(self, input: str, config: RunnableConfig | None = None) -> int:
        return len(input)
    
    async def ainvoke(self, input: str, config: RunnableConfig | None = None) -> int:
        return len(input)
    
    def stream(self, input: str, config: RunnableConfig | None = None) -> Iterator[int]:
        # For streaming support, implement transform
        for char in input:
            yield 1
    
    async def astream(self, input: str, config: RunnableConfig | None = None):
        for char in input:
            yield 1

# Immediately composable
chain = CustomRunnable() | another_step
```

Custom Runnables are automatically compatible with all composition operators.

## Summary Table

### Composition Operators

| Operator | Effect | Example |
|----------|--------|---------|
| `\|` | Sequential chaining | `step1 \| step2` |
| Dict in sequence | Parallel branching | `step1 \| {key1: step2, key2: step3}` |
| `RunnableBranch` | Conditional routing | `RunnableBranch((cond, runnable), default)` |
| `RouterRunnable` | Key-based routing | `RouterRunnable({"key": runnable})` |
| `.batch()` / `.abatch()` | Parallel input processing | `chain.batch([in1, in2])` |
| `.stream()` / `.astream()` | Token-by-token output | `for chunk in chain.stream(input):` |

### Error Handling & Resilience

| Operator | Effect | Example |
|----------|--------|---------|
| `.with_retry()` | Automatic retry with exponential backoff | `llm.with_retry(retry_if_exception_type=(TimeoutError,))` |
| `.with_fallbacks()` | Fallback to alternative Runnables on failure | `llm.with_fallbacks([fallback_llm])` |

### Serialization

| Function | Effect | Example |
|----------|--------|---------|
| `dumpd()` | Serialize to dict | `dumpd(chain)` |
| `dumps()` | Serialize to JSON string | `dumps(chain, pretty=True)` |
| `load()` | Deserialize from dict | `load(chain_dict)` |
| `loads()` | Deserialize from JSON string | `loads(chain_json)` |

See the [Runnables](runnables.md) page for protocol details and method signatures.
