# LangChain Tool Types

## `BaseTool` (base.py:392)

The **abstract base class** that all tools inherit from. It:

- Defines the core interface (`_run`, `_arun`, `invoke`, `run`, etc.)
- Handles callbacks, validation, error handling, and response formatting
- Requires subclasses to implement `_run()` method

You subclass `BaseTool` directly when you need full control over tool behavior:

```python
class MyCustomTool(BaseTool):
    name: str = "my_tool"
    description: str = "Does something"

    def _run(self, query: str, **kwargs) -> str:
        return f"Result for {query}"
```

## `Tool` (simple.py:30)

A **legacy/simple tool** for single-input functions. Key characteristics:

- Only accepts **one input argument** (string)
- If you pass multiple args, it raises `ToolException` suggesting `StructuredTool` instead
- Simpler but less flexible

```python
tool = Tool(
    name="search",
    func=lambda x: f"searched: {x}",
    description="Search for something"
)
```

## `StructuredTool` (structured.py:39)

A tool that accepts **multiple named arguments** with automatic schema inference:

- Generates a Pydantic `args_schema` from function signature
- Supports complex input types
- The modern/preferred approach for multi-argument tools

```python
tool = StructuredTool.from_function(
    func=lambda a: int, b: int: a + b,
    name="add",
    description="Add two numbers"
)
```

## `@tool` decorator (convert.py:72)

A **convenience decorator** that automatically creates the right tool type:

- If `infer_schema=True` (default) -> creates `StructuredTool`
- If `infer_schema=False` -> creates `Tool` (single string input)

```python
@tool
def search(query: str, max_results: int = 10) -> str:
    """Search the web."""
    return f"Results for {query}"
# Creates a StructuredTool with args_schema inferred from signature
```

## Summary

| Type | Input | Use Case |
|------|-------|----------|
| `BaseTool` | Any (you define) | Full customization, complex tools |
| `Tool` | Single string | Legacy, simple string-in/string-out |
| `StructuredTool` | Multiple typed args | Modern tools with schema |
| `@tool` decorator | Multiple typed args | Quick tool creation from functions |

**Recommendation**: Use the `@tool` decorator for most cases - it's the simplest and infers schema automatically.
