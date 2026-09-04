---
type: "Reference"
title: "Dict syntax creates a RunnableParallel"
openwiki_generated: true
verified:
  - by: openwiki/0.5.0
    at: 2026-09-03T15:18:34.589Z
sources:
  - id: openwiki-source-a1981e868973f6fd7f71e12e
    resource: repo://libs/core/langchain_core/runnables/base.py
  - id: openwiki-source-48e94bbe49ab4f33ba87e9cb
    resource: repo://libs/core/langchain_core/runnables/branch.py
  - id: openwiki-source-de6c904bd0171642bd50f6d9
    resource: repo://libs/core/langchain_core/runnables/router.py
generated: { by: "openwiki/0.5.0", at: "2026-09-03T15:18:34.589Z" }
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

A **`RunnableBranch`** evaluates conditions in order and executes the first matching branch:

```python
from langchain_core.runnables import RunnableBranch, RunnableLambda

branch = RunnableBranch(
    (lambda x: isinstance(x, int), RunnableLambda(lambda x: x * 2)),
    (lambda x: isinstance(x, str), RunnableLambda(lambda x: x.upper())),
    RunnableLambda(lambda x: "unknown"),
)

branch.invoke(5)        # 10
branch.invoke("hello")  # "HELLO"
branch.invoke(None)     # "unknown"
```

Conditions are evaluated sequentially; the first truthy result selects its corresponding Runnable. If no condition matches, the default branch executes.

### RouterRunnable: Key-Based Routing

A **`RouterRunnable`** routes based on a string key in the input:

```python
from langchain_core.runnables.router import RouterRunnable

add = RunnableLambda(lambda x: x + 1)
square = RunnableLambda(lambda x: x ** 2)

router = RouterRunnable(runnables={"add": add, "square": square})
router.invoke({"key": "square", "input": 3})  # 9
router.invoke({"key": "add", "input": 3})     # 4
```

The input is a dict with `"key"` (which Runnable to route to) and `"input"` (the data).

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

## Serialization and Debugging

Composed chains support serialization via the LangChain serialization system, enabling:
- **Persistence**: Save and load chains
- **Tracing**: Automatic callback integration for debugging via LangSmith
- **Inspection**: Use `get_graph()` to visualize chain structure

Enable debug output:

```python
from langchain_core.globals import set_debug

set_debug(True)  # Print intermediate results
chain.invoke(input)

# Or use callbacks:
from langchain_core.tracers import ConsoleCallbackHandler

chain.invoke(input, config={"callbacks": [ConsoleCallbackHandler()]})
```

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

| Operator | Effect | Example |
|----------|--------|---------|
| `\|` | Sequential chaining | `step1 \| step2` |
| Dict in sequence | Parallel branching | `step1 \| {key1: step2, key2: step3}` |
| `RunnableBranch` | Conditional routing | `RunnableBranch((cond, runnable), default)` |
| `RouterRunnable` | Key-based routing | `RouterRunnable({"key": runnable})` |
| `.batch()` / `.abatch()` | Parallel input processing | `chain.batch([in1, in2])` |
| `.stream()` / `.astream()` | Token-by-token output | `for chunk in chain.stream(input):` |

See the [Runnables](runnables.md) page for protocol details and method signatures.
