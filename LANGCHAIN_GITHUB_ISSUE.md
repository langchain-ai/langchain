# LangChain Bug Report - GitHub Issue Template

Copy and paste the content below directly into GitHub Issues at https://github.com/langchain-ai/langchain/issues

---

## Title

```
[Bug] AgentMiddleware lacks async support - NotImplementedError in awrap_model_call for some built-in middleware (1.0.0a14)
```

---

## Example Code

```python
from langchain.agents import create_agent
from langchain.agents.middleware import PlanningMiddleware
from langchain_anthropic import ChatAnthropic

# Create agent with middleware
model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
agent = create_agent(
    model=model,
    system_prompt="You are a helpful assistant",
    tools=[],
    middleware=[PlanningMiddleware()],
)

# Try to invoke asynchronously
config = {"configurable": {"thread_id": "test"}}
await agent.ainvoke({"messages": [{"role": "user", "content": "Hello"}]}, config)
```

---

## Error Message and Stack Trace

```
NotImplementedError: Asynchronous implementation of awrap_model_call is not available.
You are likely encountering this error because you defined only the sync version (wrap_model_call)
and invoked your agent in an asynchronous context (e.g., using `astream()` or `ainvoke()`).
To resolve this, either:
(1) subclass AgentMiddleware and implement the asynchronous awrap_model_call method,
(2) use the @wrap_model_call decorator on a standalone async function, or
(3) invoke your agent synchronously using `stream()` or `invoke()`.
```

---

## Description

### Problem Summary

Several built-in middleware classes in **LangChain 1.0.0a14** do not implement `awrap_model_call()`, which causes `NotImplementedError` when agents are invoked asynchronously using methods like `ainvoke()`, `astream()`, or `astream_events()`.

### Affected Middleware

**❌ Async NOT Supported** (NotImplementedError):
- `PlanningMiddleware`
- `AnthropicPromptCachingMiddleware`
- `ModelFallbackMiddleware`

**✅ Async Supported** (Working correctly):
- `PIIMiddleware`
- `SummarizationMiddleware`
- `HumanInTheLoopMiddleware`

**Note**: Any custom middleware inheriting from `AgentMiddleware` without implementing `awrap_model_call()` will also fail in async contexts.

### Root Cause Analysis

All middleware classes inherit the `awrap_model_call()` method from the base `AgentMiddleware` class, which only raises `NotImplementedError`:

```python
# From langchain.agents.middleware.types.AgentMiddleware
async def awrap_model_call(
    self,
    request: ModelRequest,
    handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
) -> ModelCallResult:
    msg = (
        "Asynchronous implementation of awrap_model_call is not available. "
        "You are likely encountering this error because you defined only the sync version "
        "(wrap_model_call) and invoked your agent in an asynchronous context..."
    )
    raise NotImplementedError(msg)
```

**Some middleware classes** (`PlanningMiddleware`, `AnthropicPromptCachingMiddleware`, `ModelFallbackMiddleware`) do not override this method, while others (`PIIMiddleware`, `SummarizationMiddleware`, `HumanInTheLoopMiddleware`) have properly implemented async support.

### Verification Steps

We've tested all built-in middleware with actual async agent invocations:

```python
import asyncio
from langchain.agents import create_agent
from langchain.agents.middleware import PlanningMiddleware
from langchain_anthropic import ChatAnthropic

async def test_middleware():
    model = ChatAnthropic(model="claude-3-5-sonnet-20241022")

    # Test PlanningMiddleware (will fail)
    agent = create_agent(
        model=model,
        system_prompt="You are a helpful assistant",
        tools=[],
        middleware=[PlanningMiddleware()],
    )

    config = {"configurable": {"thread_id": "test"}}
    # This will raise NotImplementedError
    await agent.ainvoke({"messages": [{"role": "user", "content": "Hello"}]}, config)

asyncio.run(test_middleware())
```

**Test Results:**

- ✅ `PIIMiddleware`: Works with async
- ✅ `SummarizationMiddleware`: Works with async
- ✅ `HumanInTheLoopMiddleware`: Works with async
- ❌ `PlanningMiddleware`: NotImplementedError
- ❌ `AnthropicPromptCachingMiddleware`: NotImplementedError
- ❌ `ModelFallbackMiddleware`: NotImplementedError

### Impact on Real-World Use Cases

This issue affects all modern async Python applications:
- ✅ **WebSocket streaming**: Real-time agent responses in chat applications
- ✅ **SSE (Server-Sent Events)**: Server-side streaming for progressive responses
- ✅ **FastAPI async endpoints**: Modern async web frameworks (FastAPI, Starlette)
- ✅ **DeepAgents**: Async multi-agent orchestration patterns
- ✅ **LangGraph async workflows**: State machine execution with async tools

### Expected Behavior

Middleware should work seamlessly in both sync and async contexts:

```python
# Both should work:
result = agent.invoke({"messages": [...]}, config)        # Sync ✅
result = await agent.ainvoke({"messages": [...]}, config) # Async ❌ (currently broken)
```

### Actual Behavior

Using the three affected middleware classes (`PlanningMiddleware`, `AnthropicPromptCachingMiddleware`, `ModelFallbackMiddleware`) with async agent methods raises `NotImplementedError`, forcing developers to:

1. Abandon those specific middleware entirely, or
2. Downgrade to synchronous agent execution (losing async benefits)

### Current Workaround

We've temporarily disabled the problematic middleware in our production environment:

```python
# Temporary workaround: Comment out unsupported middleware
agent = create_agent(
    model=model,
    system_prompt="You are a helpful assistant",
    tools=[],
    middleware=[
        # PlanningMiddleware(),  # ❌ Disabled - async not supported
        # AnthropicPromptCachingMiddleware(),  # ❌ Disabled - async not supported
        PIIMiddleware(pii_type="email"),  # ✅ Works with async
        SummarizationMiddleware(model=model),  # ✅ Works with async
    ],
)
```

We've preserved the original middleware configuration in code comments for easy re-enablement once this issue is resolved.

### Suggested Fix

Implement `awrap_model_call()` for the three affected middleware classes (`PlanningMiddleware`, `AnthropicPromptCachingMiddleware`, `ModelFallbackMiddleware`). Here's a conceptual example:

```python
class PlanningMiddleware(AgentMiddleware):
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        # Existing sync implementation
        ...

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        # Async implementation with similar logic
        # Replace sync operations with async equivalents
        response = await handler(request)
        # ... rest of async logic
        return ModelCallResult(response=response)
```

### Additional Context

- This worked correctly in earlier LangChain versions with synchronous agent execution
- The issue emerged with the async-first agent APIs introduced in the 1.0.0 alpha series
- Our production stack: **FastAPI + LangGraph + DeepAgents** with WebSocket/SSE streaming
- We're running `deepagents` which uses `async_create_deep_agent(..., is_async=True)`
- We've verified through comprehensive testing that 3 out of 6 built-in middleware lack async support
- Good news: `PIIMiddleware`, `SummarizationMiddleware`, and `HumanInTheLoopMiddleware` already work correctly with async!

---

## System Info

```
System Information
------------------
> OS:  Linux
> OS Version:  #1 SMP PREEMPT_DYNAMIC Thu Jun  5 18:30:46 UTC 2025
> Python Version:  3.12.3 (main, Aug 14 2025, 17:47:21) [GCC 13.3.0]

Package Information
-------------------
> langchain_core: 1.0.0a8
> langchain: 1.0.0a14
> langchain_community: 1.0.0a1
> langchain_anthropic: 1.0.0a4
> langchain_aws: 1.0.0a1
> langchain_google_genai: 3.0.0a1
> langchain_google_vertexai: 2.1.2
> langchain_openai: 1.0.0a4
> langgraph: 1.0.0a4
> langsmith: 0.4.35rc1
> pydantic: 2.12.0
```

---

## Checklist

- [x] I have searched the existing issues to ensure this is not a duplicate
- [x] I have provided a minimal, reproducible example
- [x] I have included the error message and stack trace
- [x] I have included my system information

---

## Closing Remarks

Thank you so much for your incredible work on LangChain! We're genuinely excited about the 1.0 release and the powerful capabilities it brings to the AI development community.

We're happy to:
- Help test any fixes you implement
- Contribute pull requests if that would be helpful
- Provide additional context or reproduction cases if needed

This is an alpha release and we completely understand that issues like this are expected. We appreciate your dedication to making LangChain the best framework for building AI applications! 🙏

Looking forward to seeing async middleware support in a future release!

---

**Helpful Links:**
- LangChain Documentation: https://python.langchain.com/docs/
- Agent Middleware Guide: https://python.langchain.com/docs/how_to/custom_agent_middleware/
- GitHub Repository: https://github.com/langchain-ai/langchain
