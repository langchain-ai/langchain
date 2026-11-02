---
type: Workflow
title: "Composability: Building Modular Systems"
description: "Composition patterns for building complex applications from simple, reusable components. Covers Runnable piping, component interfaces, and composability boundaries."
tags: [composition, runnables, lcel, modular, piping, parallel, branching, fallback]
verified:
  - by: openwiki/0.5.0
    at: 2026-10-25T08:23:03.589Z
sources:
  - id: openwiki-source-65071982a626569c8820a34b
    resource: repo://libs/core/langchain_core/runnables/__init__.py
  - id: openwiki-source-a1981e868973f6fd7f71e12e
    resource: repo://libs/core/langchain_core/runnables/base.py
  - id: openwiki-source-48e94bbe49ab4f33ba87e9cb
    resource: repo://libs/core/langchain_core/runnables/branch.py
  - id: openwiki-source-079792f059657900794e2955
    resource: repo://libs/core/langchain_core/runnables/config.py
  - id: openwiki-source-f9f4c1dc4f9cdf80d824ce15
    resource: repo://libs/core/langchain_core/runnables/fallbacks.py
generated: { by: "openwiki/0.5.0", at: "2026-10-25T08:23:03.589Z" }
---

# Composability: Building Modular Systems

LangChain's composition system enables building complex applications from simple, single-purpose components. Every component—prompts, models, parsers, tools, and custom logic—is a **Runnable**, unified by a composable interface. This document explains composition operators, component design, practical patterns, and boundaries for building production systems.

## Philosophy: Small, Single-Purpose Components

The composability philosophy rests on three principles:

1. **Single responsibility**: Each component does one thing well—transform input to output.
2. **Reusability**: Components are generic over input/output types and can be composed in any order.
3. **Declarative assembly**: Complex chains are built by composing simple operators (pipe, parallel, branch, fallback) without custom glue code.

A **Runnable** is a unit of work with a unified interface:

```python
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Each is a Runnable: prompt, model, parser
prompt = PromptTemplate.from_template("Explain {topic} in one sentence")
model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# Compose into a chain via operators
chain = prompt | model | parser

# Unified execution: sync, async, batch, stream
result = chain.invoke({"topic": "machine learning"})
```

## Input/Output Type Contracts

Every Runnable is generic over input and output types: `Runnable[Input, Output]`. Type contracts enable validation, schema introspection, and composability guarantees.

### Type Inference and Validation

Runnables infer input/output types from generic parameters or function signatures:

```python
from langchain_core.runnables import RunnableLambda
from typing import Any

def extract_keywords(text: str) -> list[str]:
    """Extract keywords from text."""
    return text.lower().split()

runnable = RunnableLambda(extract_keywords)
print(runnable.input_schema)   # Schema for str input
print(runnable.output_schema)  # Schema for list[str] output
```

When composing, the output type of one component must be compatible with the input type of the next:

```python
# Type-safe: output of prompt (str) matches model input
chain = prompt | model  # PromptTemplate[dict, str] | ChatOpenAI[str, BaseMessage]

# If types don't match, coercion happens automatically via RunnableLambda wrapping
# or an explicit adapter is needed
def adapt_to_dict(output: str) -> dict:
    return {"result": output}

adapted_chain = chain | RunnableLambda(adapt_to_dict)
```

### Schema Introspection

Composition maintains transparent schemas throughout the chain:

```python
chain = prompt | model | parser

# Input schema: what prompt expects
print(chain.input_schema)   # {"topic": str, ...}

# Output schema: what parser produces
print(chain.output_schema)  # str

# Config schema: what configuration is available
print(chain.config_schema())  # Configurable fields, if any
```

Schemas enable:
- **API documentation**: Auto-generate OpenAPI/GraphQL schemas.
- **Validation**: Reject invalid inputs before execution.
- **Tool calling**: Pass schema to LLMs for tool invocation.

## Composition Operators

### Sequential Composition: Pipe Operator `|`

The **pipe operator** chains Runnables sequentially. Each output becomes the next input.

**Location**: `repo://libs/core/langchain_core/runnables/base.py#L648-L722`

```python
from langchain_core.runnables import RunnableLambda, RunnableSequence

def add_one(x: int) -> int:
    return x + 1

def mul_two(x: int) -> int:
    return x * 2

# Pipe operator
sequence = RunnableLambda(add_one) | RunnableLambda(mul_two)

# Equivalent explicit form
sequence = RunnableSequence(
    first=RunnableLambda(add_one),
    last=RunnableLambda(mul_two),
)

sequence.invoke(5)  # (5 + 1) * 2 = 12
```

**Key behaviors**:

- **Flattening**: Nested sequences are automatically flattened to avoid redundant nesting.
- **Type propagation**: Input type from first component, output type from last.
- **Config threading**: Configuration is merged and propagated through all steps.
- **Execution modes**: Inherits sync (`invoke`), async (`ainvoke`), batch (`batch`, `abatch`), and streaming (`stream`, `astream`) from components.

**Real-world example: Prompt → Model → Parser**

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class Article(BaseModel):
    title: str = Field(..., description="Article title")
    summary: str = Field(..., description="2-3 sentence summary")

prompt = ChatPromptTemplate.from_template(
    "Write an article about {topic}. Format as JSON."
)
model = ChatOpenAI(model="gpt-4o-mini")
parser = JsonOutputParser(pydantic_object=Article)

chain = prompt | model | parser

result = chain.invoke({"topic": "quantum computing"})
# result is Article(title=..., summary=...)
```

### Parallel Composition: RunnableParallel

**Dict literal** or **RunnableParallel** invokes multiple Runnables concurrently with the same input, collecting results into a dict.

**Location**: `repo://libs/core/langchain_core/runnables/base.py#L3868-L3950`

```python
from langchain_core.runnables import RunnableParallel, RunnableLambda

# Via dict literal (automatic coercion)
parallel = RunnableLambda(lambda x: x + 1) | {
    "add": RunnableLambda(lambda x: x + 10),
    "mul": RunnableLambda(lambda x: x * 2),
}

result = parallel.invoke(5)
# {"add": 16, "mul": 12}

# Explicit RunnableParallel
parallel = RunnableParallel(
    sentiment=sentiment_chain,
    entities=entity_chain,
    topics=topic_chain,
)

result = parallel.invoke("Apple released a new iPhone today.")
# {"sentiment": "positive", "entities": ["Apple", "iPhone"], "topics": ["tech"]}
```

**Key behaviors**:

- **Same input**: All branches receive the same input simultaneously.
- **Parallel execution**: Async branches use `asyncio.gather`; sync uses thread pool.
- **Concurrency control**: `max_concurrency` in config limits parallelism.
- **Exception handling**: First exception raises unless `return_exceptions=True`.
- **Output structure**: Results are a dict with user-specified keys.

**Use cases**:

- **Multi-aspect analysis**: Sentiment, entities, topics on same text.
- **Ensemble methods**: Multiple classifiers voting on same input.
- **Context retrieval**: Fetch from multiple sources in parallel.
- **A/B testing**: Run model variants concurrently and compare.

**Real-world example: Multi-step retrieval and summarization**

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Retrieve context from multiple sources, summarize key points
parallel = {
    "input": RunnablePassthrough(),  # Pass original input through
    "context": retriever,            # Fetch relevant docs
    "entities": entity_extractor,    # Extract entities
}

chain = parallel | summary_model
```

### Branching: RunnableBranch

**RunnableBranch** evaluates conditions and routes execution to one of several branches.

**Location**: `repo://libs/core/langchain_core/runnables/branch.py#L43-L290`

```python
from langchain_core.runnables import RunnableBranch, RunnableLambda

# Multi-branch based on input type and content
branch = RunnableBranch(
    (
        lambda x: isinstance(x, str) and len(x) < 10,
        RunnableLambda(lambda x: f"Short: {x}"),
    ),
    (
        lambda x: isinstance(x, str) and len(x) >= 10,
        RunnableLambda(lambda x: f"Long: {x}"),
    ),
    RunnableLambda(lambda x: f"Not a string: {x}"),  # Default
)

branch.invoke("hi")        # "Short: hi"
branch.invoke("hello world")  # "Long: hello world"
branch.invoke(42)          # "Not a string: 42"
```

**Key behaviors**:

- **Sequential evaluation**: Conditions are checked in order; first match is selected.
- **Async conditions**: Conditions can be async or sync callables/Runnables.
- **Type-based routing**: Common pattern is `isinstance` checks for polymorphism.
- **Default fallback**: If no condition matches, the default branch executes.
- **Callback tagging**: Conditions tagged as `condition:N`, branches as `branch:N` or `branch:default`.

**Real-world example: Query classification and routing**

```python
from langchain_core.runnables import RunnableBranch

# Route queries to specialized chains
def is_math_query(x: dict) -> bool:
    return "calculate" in x["query"].lower()

def is_search_query(x: dict) -> bool:
    return "find" in x["query"].lower()

router = RunnableBranch(
    (is_math_query, math_solver_chain),
    (is_search_query, search_chain),
    general_qa_chain,  # Default for unclassified
)

result = router.invoke({"query": "calculate 2 + 2"})  # Uses math_solver_chain
result = router.invoke({"query": "find restaurants near me"})  # Uses search_chain
```

### Fallback and Retry: Resilience Patterns

#### Fallback: RunnableWithFallbacks

**RunnableWithFallbacks** tries a primary Runnable, then fallbacks in order until one succeeds.

**Location**: `repo://libs/core/langchain_core/runnables/fallbacks.py#L37-L590`

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Primary model, then fallback providers
model = ChatOpenAI(model="gpt-4o-mini").with_fallbacks([
    ChatAnthropic(model="claude-3-5-sonnet"),
    # Additional fallbacks as needed
])

result = model.invoke("Hello")
# Tries OpenAI first; if it fails, tries Anthropic
```

**Key behaviors**:

- **Sequential trials**: Runs primary, then each fallback in order.
- **Exception filtering**: Only catches exceptions in `exceptions_to_handle` (default: all).
- **Exception injection**: If `exception_key` is set, the exception dict is passed to fallbacks as input for adaptive recovery.
- **Batch retry**: For batches, tracks per-input success/failure and retries only failed inputs.
- **Async support**: Native async execution with proper exception handling.

**Real-world example: Multi-provider LLM with adaptive fallback**

```python
from langchain_core.runnables import RunnableWithFallbacks, RunnableLambda

# Fallback with error information
def adaptive_recovery(inputs: dict) -> str:
    """Generate response based on prior error."""
    error = inputs.get("error")
    if error:
        return f"Request failed: {str(error)}. Returning cached response."
    return inputs["result"]

model = ChatOpenAI().with_fallbacks(
    [ChatAnthropic(), RunnableLambda(adaptive_recovery)],
    exception_key="error",
)
```

#### Retry: with_retry()

Retry logic is available on any Runnable via `.with_retry()`:

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini").with_retry(
    retry_if_exception_type=(RuntimeError,),
    stop_after_attempt=3,
    wait_exponential_jitter=True,
)

result = model.invoke("Hello")
# Retries up to 3 times on RuntimeError with exponential backoff
```

**Key behaviors**:

- **Selective retry**: Only retries on specified exception types.
- **Exponential backoff**: Built-in exponential backoff with optional jitter.
- **Attempt tracking**: Tags callback runs as `retry:attempt:N`.
- **Batch retry**: In batch mode, retries only failed inputs.
- **Stream limitation**: Streaming is not retried (ambiguous semantics).

## Component Interface Design

Composable components follow a consistent interface:

### 1. Single Transformation

Each component transforms a single input to a single output. No side effects in the core execution path:

```python
from langchain_core.runnables import RunnableLambda

# Good: pure transformation
def extract_text(doc: dict) -> str:
    return doc["content"]

runnable = RunnableLambda(extract_text)
```

### 2. Type Clarity

Make input and output types explicit via type hints or generic parameters:

```python
from langchain_core.runnables import Runnable
from typing import Generic

# Explicit types in function signature
def process(x: str) -> dict[str, int]:
    return {"length": len(x)}

# Or as a Runnable subclass
class CustomRunnable(Runnable[str, dict[str, int]]):
    def invoke(self, input: str, config=None):
        return {"length": len(input)}
```

### 3. Composition Boundaries

Know when to create a new component vs. extend an existing one:

| Scenario | Action | Reason |
|----------|--------|--------|
| **Reusable logic across chains** | Create a new Runnable | Promotes reuse and testing |
| **Linear sequence of transformations** | Keep as one Runnable | Reduces nesting; same transform |
| **Conditional routing within component** | Extract to RunnableBranch | Makes control flow explicit |
| **Catching specific exceptions** | Wrap with RunnableWithFallbacks | Separates error handling |
| **Simple transformations (< 5 lines)** | Use RunnableLambda | Low ceremony; easy to inline |

**Example: When to create a component**

```python
# AVOID: Logic buried in lambda
chain = prompt | model | RunnableLambda(
    lambda x: json.loads(x) if x.startswith("{") else {"error": x}
)

# BETTER: Extract to named component
class SafeJsonParser(Runnable[str, dict]):
    def invoke(self, input: str, config=None):
        try:
            return json.loads(input)
        except:
            return {"error": input}

chain = prompt | model | SafeJsonParser()
```

### 4. Configuration Support

Mark dynamic fields as configurable so they can be overridden at runtime:

```python
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini").configurable_fields(
    model=ConfigurableField(
        id="model_name",
        name="Model Name",
        description="Which model to use",
    ),
    temperature=ConfigurableField(
        id="temp",
        name="Temperature",
        description="Sampling temperature",
    ),
)

# Override at runtime without recreation
result = model.invoke(
    "Hello",
    config={"configurable": {"model_name": "gpt-4-turbo", "temp": 0.5}},
)
```

## Practical Composition Patterns

### Pattern 1: Input Enrichment

Add context before invoking a chain:

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Preserve input, fetch context in parallel
chain = {
    "input": RunnablePassthrough(),
    "context": retriever,
} | RunnableLambda(
    lambda x: f"Question: {x['input']}\nContext: {x['context']}"
) | model | parser
```

### Pattern 2: Multi-Stage Extraction

Extract different aspects in parallel, then merge:

```python
chain = (
    model
    | RunnableParallel(
        text=lambda x: x.content,
        metadata=lambda x: x.response_metadata,
        usage=lambda x: x.usage_metadata,
    )
    | RunnableLambda(
        lambda results: {
            "answer": results["text"],
            "cost": results["usage"],
            "model": results["metadata"]["model"],
        }
    )
)
```

### Pattern 3: Conditional Retry with Fallback

Retry once, then fallback:

```python
primary = model.with_retry(
    stop_after_attempt=2,
    retry_if_exception_type=(TimeoutError,),
)

fallback_chain = (
    primary.with_fallbacks([
        cached_response_provider,
    ])
)
```

### Pattern 4: Ensemble with Consensus

Run multiple models, vote on result:

```python
ensemble = RunnableParallel(
    model_a=model_a,
    model_b=model_b,
    model_c=model_c,
) | RunnableLambda(
    lambda results: majority_vote(
        results["model_a"],
        results["model_b"],
        results["model_c"],
    )
)
```

### Pattern 5: Conditional Routing Based on Input Analysis

Classify input, then route to specialized chain:

```python
classifier = RunnableLambda(classify_query)

router = RunnableBranch(
    (
        lambda x: x["category"] == "technical",
        technical_qa_chain,
    ),
    (
        lambda x: x["category"] == "creative",
        creative_chain,
    ),
    general_chain,
)

chain = classifier | router
```

## Composability Boundaries

When to stop composing and create a new abstraction:

### 1. Deep Nesting (> 5 levels)

Deep chains are hard to debug. Extract intermediate groups:

```python
# AVOID: Chain of 8+ components
chain = step1 | step2 | step3 | step4 | step5 | step6 | step7 | step8

# BETTER: Group related steps
preprocessing = step1 | step2 | step3
inference = step4 | step5
postprocessing = step6 | step7 | step8
chain = preprocessing | inference | postprocessing
```

### 2. Reusable Subchains

Subchains used in multiple places should be extracted and named:

```python
# Reused in multiple contexts
retrieval_chain = query_processor | retriever | context_formatter

answer_chain = retrieval_chain | model | parser
summary_chain = retrieval_chain | summarizer
```

### 3. Stateful or Side-Effect Components

Components with state or side effects (logging, metrics, external calls) should be isolated and clear:

```python
class LoggingRunnable(Runnable[Input, Output]):
    """Wraps a runnable and logs input/output."""
    def __init__(self, runnable, logger):
        self.runnable = runnable
        self.logger = logger

    def invoke(self, input, config=None):
        self.logger.info(f"Input: {input}")
        output = self.runnable.invoke(input, config)
        self.logger.info(f"Output: {output}")
        return output
```

## Execution Modes Across Composition

All composition operators automatically support the full execution API: sync, async, batch, and streaming.

### Sync Execution

```python
chain = prompt | model | parser
result = chain.invoke({"topic": "AI"})
```

### Async Execution

```python
result = await chain.ainvoke({"topic": "AI"})
```

### Batch Processing

```python
inputs = [{"topic": "AI"}, {"topic": "ML"}, {"topic": "DL"}]
results = chain.batch(inputs)
```

### Streaming

```python
for chunk in chain.stream({"topic": "AI"}):
    print(chunk, end="", flush=True)
```

### Async Streaming

```python
async for chunk in chain.astream({"topic": "AI"}):
    print(chunk, end="", flush=True)
```

## Configuration Threading Through Composition

Configuration (tags, metadata, callbacks, concurrency limits) flows through composed chains automatically:

```python
chain = prompt | model | parser

result = chain.invoke(
    {"topic": "AI"},
    config={
        "tags": ["production", "v1"],
        "metadata": {"user_id": "123"},
        "callbacks": [tracing_handler],
        "max_concurrency": 5,
    },
)
```

All intermediate steps receive merged config:
- Tags and metadata accumulate.
- Callbacks receive hooks for all steps.
- `max_concurrency` applies to batch and parallel operations.

## Debugging and Visualization

### Implicit Debugging

Enable global debug to see all step executions:

```python
from langchain_core.globals import set_debug

set_debug(True)
chain.invoke({"topic": "AI"})
# Prints invocation details for each step
```

### Explicit Callbacks

Use custom callbacks for fine-grained observability:

```python
from langchain_core.callbacks import BaseCallbackHandler

class MyHandler(BaseCallbackHandler):
    def on_chain_start(self, serialized, inputs, **kwargs):
        print(f"Chain started: {inputs}")

chain.invoke(
    {"topic": "AI"},
    config={"callbacks": [MyHandler()]},
)
```

### Graph Visualization

Visualize the composition graph:

```python
graph = chain.get_graph()
graph.print_ascii()  # ASCII art representation
# or
graph.draw_mermaid_png(outfile="chain.png")  # Mermaid diagram
```

## Summary: Composability Best Practices

1. **Single responsibility**: Each component does one thing.
2. **Type safety**: Explicit input/output types enable validation and composition.
3. **Declarative assembly**: Use operators (`|`, dicts, RunnableBranch) rather than custom glue.
4. **Config threading**: Leverage built-in config propagation; don't hardcode settings.
5. **Reuse over inheritance**: Compose smaller pieces rather than extending large classes.
6. **Extract boundaries**: Create named subchains for readability and reusability.
7. **Test components independently**: Unit-test individual Runnables before composition.
8. **Monitor execution**: Use callbacks and debugging to understand production behavior.
