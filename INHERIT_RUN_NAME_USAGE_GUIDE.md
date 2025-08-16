# Using `inherit_run_name` for Dynamic Run Naming in LangChain

## Overview

The `inherit_run_name` configuration option allows you to control whether run names are propagated to child runs in LangChain runnable chains. This is particularly useful for tracing and monitoring tools that need consistent naming across complex execution flows.

## Default Behavior (Backward Compatible)

By default, `run_name` is **NOT** inherited by child runs. Each child run gets its own name based on the runnable's class name or explicitly configured name.

```python
from langchain_core.runnables import RunnableLambda

def step1(x):
    return f"Step 1: {x}"

def step2(x):
    return f"Step 2: {x}"

chain = RunnableLambda(step1) | RunnableLambda(step2)

# Without inherit_run_name, only the root run has "my_chain"
# Child runs will have their default names
result = chain.invoke(
    "input",
    config={"run_name": "my_chain"}
)
```

## Enabling Run Name Inheritance

To propagate the same run name throughout the entire chain, set `inherit_run_name=True`:

```python
# With inherit_run_name=True, all runs in the chain will have "my_chain"
result = chain.invoke(
    "input",
    config={
        "run_name": "my_chain",
        "inherit_run_name": True  # Enable inheritance
    }
)
```

## Use Cases

### 1. Dynamic Tracing with Consistent Names

When using tracing tools like Langfuse, you might want all steps in a chain to share the same dynamic run name:

```python
import uuid
from datetime import datetime

# Generate a dynamic run name based on context
run_name = f"user_{user_id}_query_{datetime.now().isoformat()}"

# All steps in the chain will have this dynamic name
result = chain.invoke(
    user_query,
    config={
        "run_name": run_name,
        "inherit_run_name": True,
        "callbacks": [langfuse_handler]
    }
)
```

### 2. Per-Component Custom Names (Current Workaround)

If you need different names for different steps in your chain, use `with_config()` on individual components:

```python
# Each step has its own specific name
chain = (
    RunnableLambda(step1).with_config(run_name="data_preprocessing") |
    RunnableLambda(step2).with_config(run_name="llm_processing") |
    RunnableLambda(step3).with_config(run_name="output_formatting")
)

# Invoke without inherit_run_name
result = chain.invoke(
    "input",
    config={"run_name": "main_pipeline"}
)
```

### 3. Hybrid Approach

You can combine both approaches - use inheritance for most of the chain but override specific steps:

```python
# Most steps inherit "data_pipeline", but step2 has its own name
chain = (
    RunnableLambda(step1) |
    RunnableLambda(step2).with_config(run_name="special_processing") |
    RunnableLambda(step3)
)

result = chain.invoke(
    "input",
    config={
        "run_name": "data_pipeline",
        "inherit_run_name": True
    }
)
# Result: step1 and step3 have "data_pipeline", step2 has "special_processing"
```

## Configuration Reference

| Config Key | Type | Default | Description |
|------------|------|---------|-------------|
| `run_name` | `str` | `None` | The name for the run |
| `inherit_run_name` | `bool` | `False` | Whether to propagate run_name to child runs |

## Migration Guide

### Before (Manual Naming Each Step)
```python
# Had to manually name each step
chain = (
    RunnableLambda(step1).with_config(run_name="my_flow") |
    RunnableLambda(step2).with_config(run_name="my_flow") |
    RunnableLambda(step3).with_config(run_name="my_flow")
)
```

### After (With inherit_run_name)
```python
# Now can set once and inherit
chain = RunnableLambda(step1) | RunnableLambda(step2) | RunnableLambda(step3)

result = chain.invoke(
    "input",
    config={
        "run_name": "my_flow",
        "inherit_run_name": True
    }
)
```

## Important Notes

1. **Backward Compatibility**: The default behavior (`inherit_run_name=False`) maintains full backward compatibility with existing code.

2. **Precedence**: When `inherit_run_name=True` is set globally, it takes precedence over step-specific names unless the step explicitly sets `inherit_run_name=False`.

3. **Performance**: There is no performance impact from using `inherit_run_name`.

4. **Tracing Tools**: This feature is particularly useful with tracing tools like Langfuse, Langsmith, or custom callback handlers that need consistent run identification.

## Example: Complete Application

```python
from langchain_core.runnables import RunnableLambda
from langchain_core.callbacks.base import BaseCallbackHandler
from typing import Any, Dict
import json

class CustomTracingHandler(BaseCallbackHandler):
    """Custom handler that tracks run names for debugging."""
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        run_name = kwargs.get("name", "unnamed")
        print(f"[TRACE] Starting: {run_name}")
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        run_name = kwargs.get("name", "unnamed")
        print(f"[TRACE] Completed: {run_name}")

# Define processing steps
def extract_data(input_text: str) -> dict:
    """Extract structured data from text."""
    return {"extracted": input_text.upper()}

def enrich_data(data: dict) -> dict:
    """Enrich the extracted data."""
    data["enriched"] = True
    return data

def format_output(data: dict) -> str:
    """Format the final output."""
    return json.dumps(data, indent=2)

# Build the processing chain
processing_chain = (
    RunnableLambda(extract_data) |
    RunnableLambda(enrich_data) |
    RunnableLambda(format_output)
)

# Use with dynamic run naming
tracer = CustomTracingHandler()

# Example 1: Without inheritance (default)
print("=== Without inherit_run_name ===")
result = processing_chain.invoke(
    "sample input",
    config={
        "run_name": "etl_pipeline_v1",
        "callbacks": [tracer]
    }
)
# Only the root run will be named "etl_pipeline_v1"

print("\n=== With inherit_run_name ===")
# Example 2: With inheritance
result = processing_chain.invoke(
    "sample input",
    config={
        "run_name": "etl_pipeline_v2",
        "inherit_run_name": True,
        "callbacks": [tracer]
    }
)
# All runs in the chain will be named "etl_pipeline_v2"

print(f"\nResult: {result}")
```

## Troubleshooting

**Q: My child runs still don't have the custom name even with `inherit_run_name=True`**
A: Ensure you're using a recent version of langchain-core that includes this feature. Also verify that intermediate steps aren't explicitly overriding the configuration.

**Q: Can I use `inherit_run_name` with async chains?**
A: Yes, `inherit_run_name` works with both sync (`invoke`) and async (`ainvoke`) execution.

**Q: How does this interact with batching?**
A: When using `batch()`, each item in the batch can have its own configuration, including different `run_name` and `inherit_run_name` settings.

## Summary

- Use `inherit_run_name=True` when you want consistent run names throughout a chain
- Use per-component `with_config()` when you need different names for different steps
- The default behavior (`inherit_run_name=False`) maintains backward compatibility
- This feature is especially useful for tracing and monitoring complex LangChain applications
