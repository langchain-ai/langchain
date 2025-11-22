# MCP Stateful Session Management Guide

## Overview

The Model Context Protocol (MCP) enables LangChain agents to interact with external tools and services. By default, MCP tools use **stateless sessions**, creating a new session for each tool invocation. While this works for simple operations, it causes issues with tools that require persistent state, such as browser automation, database connections, or file system operations.

This guide explains how to use **stateful MCP sessions** to maintain persistent connections across multiple tool invocations, preventing common issues like browser session termination and "Ref not found" errors.

## Table of Contents

- [The Problem: Stateless vs Stateful Sessions](#the-problem-stateless-vs-stateful-sessions)
- [When to Use Stateful Sessions](#when-to-use-stateful-sessions)
- [Implementation Patterns](#implementation-patterns)
- [Code Examples](#code-examples)
- [Common Pitfalls and Troubleshooting](#common-pitfalls-and-troubleshooting)
- [Migration Guide](#migration-guide)
- [API Reference](#api-reference)

## The Problem: Stateless vs Stateful Sessions

### Stateless Sessions (Default Behavior)

By default, when you use MCP tools with LangChain agents, each tool invocation creates a new session:

```python
# ❌ PROBLEMATIC: Each tool call creates a new session
from langchain_mcp_adapters import MultiServerMCPClient
from langchain.agents import create_agent

mcp_client = MultiServerMCPClient()

# This creates a new session for EACH tool call
tools = await mcp_client.get_tools("playwright")
agent = create_agent(model="gpt-4", tools=tools)

# First call: Creates session A, navigates, then closes session A
result1 = agent.invoke({"messages": [{"role": "user", "content": "Navigate to example.com"}]})

# Second call: Creates session B, tries to click, but page is gone!
result2 = agent.invoke({"messages": [{"role": "user", "content": "Click the login button"}]})
# Error: "Ref not found" - the browser session was terminated!
```

**What happens:**
1. First tool call creates a browser session, navigates to the page, then closes the session
2. Second tool call creates a new browser session with no page loaded
3. Attempting to click fails because the element reference doesn't exist in the new session

### Stateful Sessions (Recommended for Persistent Tools)

Stateful sessions maintain a persistent connection across multiple tool invocations:

```python
# ✅ CORRECT: Single session maintained across all tool calls
from langchain_mcp_adapters import MultiServerMCPClient, load_mcp_tools
from langchain.agents import create_agent

mcp_client = MultiServerMCPClient()

# Create a persistent session
async with mcp_client.session("playwright") as session:
    # Load tools with the session
    tools = await load_mcp_tools(session)
    agent = create_agent(model="gpt-4", tools=tools)
    
    # All tool calls use the SAME session
    result1 = agent.invoke({"messages": [{"role": "user", "content": "Navigate to example.com"}]})
    result2 = agent.invoke({"messages": [{"role": "user", "content": "Click the login button"}]})
    # Success! The browser session persists between calls
```

## When to Use Stateful Sessions

Use stateful sessions when working with tools that maintain state between operations:

### Required for Stateful Tools

| Tool Type | Why Stateful? | Example Operations |
|-----------|--------------|-------------------|
| **Browser Automation** (Playwright, Puppeteer) | Browser windows close between calls | Navigate → Click → Type → Submit |
| **Database Connections** | Connection pools and transactions | BEGIN → INSERT → UPDATE → COMMIT |
| **File System Operations** | Working directory and file handles | Open file → Read → Modify → Save |
| **Terminal/Shell Sessions** | Environment variables and working directory | cd directory → run command → check output |
| **API Sessions** | Authentication tokens and rate limiting | Login → Make requests → Logout |

### Not Required for Stateless Tools

| Tool Type | Why Stateless Works | Example Operations |
|-----------|-------------------|-------------------|
| **Simple Calculations** | No state needed | Calculate sum, Convert units |
| **Data Transformations** | Pure functions | Parse JSON, Format text |
| **Single API Calls** | Each call is independent | Get weather, Search web |

## Implementation Patterns

### Pattern 1: Using StatefulMCPAgentExecutor (Recommended)

The `StatefulMCPAgentExecutor` class provides the simplest way to create agents with stateful MCP sessions:

```python
from langchain.agents.mcp_utils import StatefulMCPAgentExecutor
from langchain_mcp_adapters import MultiServerMCPClient

mcp_client = MultiServerMCPClient()

async with StatefulMCPAgentExecutor(
    client=mcp_client,
    server_name="playwright",
    model="gpt-4",
    system_prompt="You are a browser automation assistant.",
) as executor:
    # All invocations use the same session
    result = await executor.ainvoke({
        "messages": [{"role": "user", "content": "Navigate to example.com and click login"}]
    })
```

**Benefits:**
- Automatic session lifecycle management
- Built-in error handling and cleanup
- Support for both sync and async execution
- No manual session management needed

### Pattern 2: Using mcp_agent_session Context Manager

For simpler use cases, use the `mcp_agent_session` context manager:

```python
from langchain.agents.mcp_utils import mcp_agent_session
from langchain_mcp_adapters import MultiServerMCPClient

mcp_client = MultiServerMCPClient()

async with mcp_agent_session(
    client=mcp_client,
    server_name="playwright",
    model="gpt-4",
) as agent:
    # Use the agent with persistent session
    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "Automate browser task"}]
    })
```

### Pattern 3: Manual Session Management

For full control over session lifecycle:

```python
from langchain.agents.mcp_utils import create_stateful_mcp_agent
from langchain_mcp_adapters import MultiServerMCPClient

mcp_client = MultiServerMCPClient()

# Create agent with manual session management
agent, session = await create_stateful_mcp_agent(
    client=mcp_client,
    server_name="playwright",
    model="gpt-4",
    auto_cleanup=False,  # Manual cleanup required
)

try:
    # Use the agent
    result = await agent.ainvoke({"messages": [...]})
finally:
    # Manual cleanup
    await session.__aexit__(None, None, None)
```

### Pattern 4: Using create_agent with MCP Session Config

The standard `create_agent` function now supports MCP session configuration:

```python
from langchain.agents import create_agent
from langchain_mcp_adapters import MultiServerMCPClient

mcp_client = MultiServerMCPClient()

agent = create_agent(
    model="gpt-4",
    tools=mcp_tools,
    mcp_session_config={
        "client": mcp_client,
        "server_name": "playwright",
        "auto_cleanup": True,
    }
)
```

## Code Examples

### Example 1: Browser Automation with Playwright

```python
import asyncio
from langchain.agents.mcp_utils import StatefulMCPAgentExecutor
from langchain_mcp_adapters import MultiServerMCPClient

async def automate_login():
    """Automate a login flow using stateful Playwright session."""
    
    mcp_client = MultiServerMCPClient()
    
    async with StatefulMCPAgentExecutor(
        client=mcp_client,
        server_name="playwright",
        model="gpt-4",
        system_prompt="You are a browser automation expert. Be precise with selectors.",
    ) as executor:
        
        # Multi-step login flow - all in the same browser session
        steps = [
            "Navigate to https://example.com/login",
            "Type 'user@example.com' into the email field",
            "Type 'password123' into the password field",
            "Click the submit button",
            "Wait for the dashboard to load",
            "Click on the profile menu",
        ]
        
        for step in steps:
            result = await executor.ainvoke({
                "messages": [{"role": "user", "content": step}]
            })
            print(f"Completed: {step}")
            
            # Check for errors in the response
            if "error" in str(result).lower():
                print(f"Error detected: {result}")
                break

# Run the automation
asyncio.run(automate_login())
```

### Example 2: Database Transaction Management

```python
from langchain.agents.mcp_utils import mcp_agent_session
from langchain_mcp_adapters import MultiServerMCPClient

async def process_database_transaction():
    """Execute a database transaction with proper session management."""
    
    mcp_client = MultiServerMCPClient()
    
    async with mcp_agent_session(
        client=mcp_client,
        server_name="postgresql",
        model="gpt-4",
        system_prompt="You are a database expert. Ensure data integrity.",
    ) as agent:
        
        # Transaction operations - all in the same database session
        operations = [
            "BEGIN TRANSACTION",
            "INSERT INTO users (name, email) VALUES ('John Doe', 'john@example.com')",
            "UPDATE accounts SET balance = balance + 100 WHERE user_email = 'john@example.com'",
            "SELECT * FROM users WHERE email = 'john@example.com'",
            "COMMIT",
        ]
        
        for operation in operations:
            result = await agent.ainvoke({
                "messages": [{"role": "user", "content": f"Execute: {operation}"}]
            })
            
            # Rollback on error
            if "error" in str(result).lower() and "COMMIT" not in operation:
                await agent.ainvoke({
                    "messages": [{"role": "user", "content": "ROLLBACK"}]
                })
                raise Exception(f"Transaction failed: {result}")
```

### Example 3: Comparing Stateless vs Stateful Behavior

```python
from langchain_mcp_adapters import MultiServerMCPClient, load_mcp_tools
from langchain.agents import create_agent
import asyncio

async def demonstrate_difference():
    """Show the difference between stateless and stateful sessions."""
    
    mcp_client = MultiServerMCPClient()
    
    print("=== STATELESS (Problematic) ===")
    try:
        # Each tool call creates a new session
        tools = await mcp_client.get_tools("playwright")
        agent = create_agent(model="gpt-4", tools=tools)
        
        # Navigate - creates and destroys session A
        agent.invoke({"messages": [{"role": "user", "content": "Navigate to example.com"}]})
        
        # Click - creates session B, fails because no page loaded
        agent.invoke({"messages": [{"role": "user", "content": "Click the button"}]})
    except Exception as e:
        print(f"Stateless failed as expected: {e}")
    
    print("\n=== STATEFUL (Correct) ===")
    # Single session maintained across all calls
    async with mcp_client.session("playwright") as session:
        tools = await load_mcp_tools(session)
        agent = create_agent(model="gpt-4", tools=tools)
        
        # Navigate - uses persistent session
        agent.invoke({"messages": [{"role": "user", "content": "Navigate to example.com"}]})
        
        # Click - works because session persists
        result = agent.invoke({"messages": [{"role": "user", "content": "Click the button"}]})
        print(f"Stateful succeeded: {result}")

asyncio.run(demonstrate_difference())
```

## Common Pitfalls and Troubleshooting

### Pitfall 1: Forgetting to Use Stateful Sessions

**Symptom:** "Ref not found" errors, browser closing between operations

**Solution:** Always use stateful sessions for browser automation:
```python
# ❌ Wrong
tools = await mcp_client.get_tools("playwright")

# ✅ Correct
async with mcp_client.session("playwright") as session:
    tools = await load_mcp_tools(session)
```

### Pitfall 2: Not Handling Session Cleanup

**Symptom:** Resource leaks, hanging browser processes

**Solution:** Always use context managers or ensure manual cleanup:
```python
# ✅ Automatic cleanup with context manager
async with StatefulMCPAgentExecutor(...) as executor:
    # Session cleaned up automatically

# ✅ Manual cleanup if not using context manager
try:
    agent, session = await create_stateful_mcp_agent(...)
    # Use agent
finally:
    await session.__aexit__(None, None, None)
```

### Pitfall 3: Mixing Stateful and Stateless Tools

**Symptom:** Inconsistent behavior, some operations work while others fail

**Solution:** Group tools by session requirements:
```python
# Separate stateful and stateless tools
stateful_tools = await load_mcp_tools(session)  # Browser, database
stateless_tools = [calculate_tool, format_tool]  # Simple utilities

all_tools = stateful_tools + stateless_tools
agent = create_agent(model="gpt-4", tools=all_tools)
```

### Pitfall 4: Session Timeout

**Symptom:** Operations fail after long idle periods

**Solution:** Implement session refresh or reconnection logic:
```python
async def with_session_refresh(executor, message, max_retries=2):
    """Execute with automatic session refresh on timeout."""
    for attempt in range(max_retries):
        try:
            return await executor.ainvoke({"messages": [message]})
        except SessionTimeoutError:
            if attempt < max_retries - 1:
                # Recreate session
                await executor.refresh_session()
            else:
                raise
```

## Migration Guide

### Step 1: Identify Affected Code

Look for patterns that indicate stateless MCP usage:

```python
# Signs of stateless usage:
tools = await mcp_client.get_tools("playwright")
tools = mcp_client.list_tools()
agent = create_agent(model="...", tools=tools)
```

### Step 2: Choose Migration Pattern

Based on your use case:

1. **Simple scripts** → Use `mcp_agent_session` context manager
2. **Complex workflows** → Use `StatefulMCPAgentExecutor`
3. **Existing create_agent calls** → Add `mcp_session_config` parameter
4. **Full control needed** → Use `create_stateful_mcp_agent` with manual management

### Step 3: Update Your Code

**Before (Stateless):**
```python
from langchain_mcp_adapters import MultiServerMCPClient
from langchain.agents import create_agent

mcp_client = MultiServerMCPClient()
tools = await mcp_client.get_tools("playwright")
agent = create_agent(model="gpt-4", tools=tools)

# Multiple invocations fail
for task in tasks:
    result = agent.invoke({"messages": [{"role": "user", "content": task}]})
```

**After (Stateful):**
```python
from langchain.agents.mcp_utils import StatefulMCPAgentExecutor
from langchain_mcp_adapters import MultiServerMCPClient

mcp_client = MultiServerMCPClient()

async with StatefulMCPAgentExecutor(
    client=mcp_client,
    server_name="playwright",
    model="gpt-4",
) as executor:
    # Multiple invocations succeed
    for task in tasks:
        result = await executor.ainvoke({"messages": [{"role": "user", "content": task}]})
```

### Step 4: Test Your Changes

Verify that:
1. Multi-step workflows complete successfully
2. No "Ref not found" or session termination errors
3. Resources are properly cleaned up
4. Performance is acceptable (session creation overhead is minimized)

## API Reference

### StatefulMCPAgentExecutor

```python
class StatefulMCPAgentExecutor:
    """Executor that maintains a stateful MCP session for agents."""
    
    def __init__(
        self,
        client: Any,  # MCP client instance
        server_name: str,  # MCP server name
        model: str | BaseChatModel,  # LLM model
        system_prompt: str | None = None,  # Optional system prompt
        tools: Sequence[BaseTool] | None = None,  # Additional tools
    ):
        """Initialize the executor with MCP session configuration."""
    
    async def ainvoke(self, input: dict, config: dict | None = None, **kwargs) -> dict:
        """Asynchronously invoke the agent."""
    
    def invoke(self, input: dict, config: dict | None = None, **kwargs) -> dict:
        """Synchronously invoke the agent."""
    
    async def astream(self, input: dict, config: dict | None = None, **kwargs):
        """Stream agent responses asynchronously."""
```

### create_stateful_mcp_agent

```python
async def create_stateful_mcp_agent(
    client: Any,  # MCP client instance
    server_name: str,  # MCP server name
    model: str | BaseChatModel,  # LLM model
    system_prompt: str | None = None,  # Optional system prompt
    tools: Sequence[BaseTool] | None = None,  # Additional tools
    auto_cleanup: bool = True,  # Auto cleanup on deletion
) -> tuple[Any, Any]:  # Returns (agent, session)
    """Create an agent with stateful MCP session."""
```

### mcp_agent_session

```python
@asynccontextmanager
async def mcp_agent_session(
    client: Any,  # MCP client instance
    server_name: str,  # MCP server name
    model: str | BaseChatModel,  # LLM model
    system_prompt: str | None = None,  # Optional system prompt
    tools: Sequence[BaseTool] | None = None,  # Additional tools
) -> Any:  # Yields agent
    """Context manager for creating agents with automatic session management."""
```

### MCPSessionConfig

```python
class MCPSessionConfig(TypedDict):
    """Configuration for MCP session management in create_agent."""
    
    client: Required[Any]  # MCP client instance
    server_name: Required[str]  # MCP server name
    auto_cleanup: NotRequired[bool]  # Auto cleanup (default: True)
```

## Best Practices

1. **Always use context managers** for automatic resource cleanup
2. **Group related operations** within the same session context
3. **Handle errors gracefully** with proper rollback/cleanup logic
4. **Monitor session lifetime** for long-running processes
5. **Document session requirements** in your tool descriptions
6. **Test both success and failure paths** with session management
7. **Use appropriate session scope** (don't keep sessions open unnecessarily)

## Conclusion

Stateful MCP session management is essential for tools that maintain state between operations. By using the patterns and utilities described in this guide, you can avoid common issues like browser session termination and ensure reliable multi-step workflows.

For more information, see:
- [LangChain Agents Documentation](https://docs.langchain.com/docs/concepts/agents)
- [Model Context Protocol Specification](https://github.com/modelcontextprotocol/specification)
- [LangChain MCP Adapters](https://github.com/langchain-ai/langchain-mcp-adapters)
