"""Example demonstrating stateful MCP usage with Playwright browser automation.

This example shows how to properly maintain browser sessions across multiple tool calls
using the MCP (Model Context Protocol) client with stateful session management.

The key difference between stateless and stateful MCP usage:
- Stateless (default): Each tool call creates a new session, browser closes after each call
- Stateful (recommended for browser automation): Single session maintained across all tool calls

This solves the common issue where browser sessions terminate immediately after navigation,
causing subsequent tool calls to fail with "Ref not found" errors.
"""

import asyncio

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from langchain.agents import create_agent

# Note: langchain-mcp-adapters must be installed separately
# pip install langchain-mcp-adapters
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools
except ImportError:
    raise ImportError("Please install langchain-mcp-adapters: pip install langchain-mcp-adapters")


async def stateless_example_problematic():
    """Problematic example showing stateless MCP usage that causes browser session termination.

    This demonstrates the INCORRECT way that leads to the reported issue.
    Each tool call creates a new browser session, causing state loss between calls.
    """
    print("\n" + "=" * 80)
    print("PROBLEMATIC EXAMPLE: Stateless MCP Usage (Browser closes after each tool call)")
    print("=" * 80)

    # Initialize MCP client with Playwright server
    client = MultiServerMCPClient(
        {
            "playwright": {
                "command": "npx",
                "args": [
                    "@playwright/mcp@latest",
                    "--headless",  # Run in headless mode
                    "--isolated",  # Allow multiple browser instances
                ],
                "transport": "stdio",
            }
        }
    )

    # PROBLEM: Using get_tools() creates stateless tools
    # Each tool invocation will create a new session
    tools = await client.get_tools()
    print(f"✗ Loaded {len(tools)} stateless MCP tools")

    # Create model and agent
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(model="gpt-4", temperature=0)
    model_with_tools = model.bind_tools(tools)

    # Create agent with memory
    memory = MemorySaver()
    agent = create_agent(
        model_with_tools,
        tools,
        debug=True,
        system_prompt="You are a web testing engineer using Playwright tools.",
        checkpointer=memory,
    )

    # This will fail on the second tool call because the browser session is lost
    test_prompt = """
    1. Navigate to https://example.com
    2. Take a screenshot
    3. Click on the 'More information...' link
    """

    try:
        config = {"configurable": {"thread_id": "test-thread"}}
        response = await agent.ainvoke(
            {"messages": [HumanMessage(content=test_prompt)]},
            config=config,
        )
        print("Response:", response["messages"][-1].content)
    except Exception as e:
        print(f"❌ Error (expected): {e}")
        print("This error occurs because the browser session was terminated after navigation")


async def stateful_example_correct():
    """Correct example showing stateful MCP usage that maintains browser sessions.

    This demonstrates the CORRECT way to use MCP tools for browser automation.
    A single browser session is maintained across all tool calls.
    """
    print("\n" + "=" * 80)
    print("CORRECT EXAMPLE: Stateful MCP Usage (Browser session persists)")
    print("=" * 80)

    # Initialize MCP client with Playwright server
    client = MultiServerMCPClient(
        {
            "playwright": {
                "command": "npx",
                "args": [
                    "@playwright/mcp@latest",
                    "--headless",  # Run in headless mode
                    "--isolated",  # Allow multiple browser instances
                ],
                "transport": "stdio",
            }
        }
    )

    # SOLUTION: Create a persistent session and load tools from it
    # This maintains the browser session across all tool calls
    async with client.session("playwright") as session:
        # Load tools with the persistent session
        tools = await load_mcp_tools(session)
        print(f"✓ Loaded {len(tools)} stateful MCP tools with persistent session")

        # Create model and agent
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model="gpt-4", temperature=0)
        model_with_tools = model.bind_tools(tools)

        # Create agent with memory
        memory = MemorySaver()
        agent = create_agent(
            model_with_tools,
            tools,
            debug=True,
            system_prompt="You are a web testing engineer using Playwright tools.",
            checkpointer=memory,
        )

        # Complex multi-step browser interaction that requires session persistence
        test_prompt = """
        Please perform the following browser automation tasks:
        1. Navigate to https://example.com
        2. Take a screenshot of the page
        3. Click on the 'More information...' link
        4. Take another screenshot after navigation
        5. Go back to the previous page
        6. Verify you're back on example.com
        """

        try:
            config = {"configurable": {"thread_id": "test-thread"}}
            response = await agent.ainvoke(
                {"messages": [HumanMessage(content=test_prompt)]},
                config=config,
            )
            print("✓ Success! All browser operations completed in the same session")
            print("Response:", response["messages"][-1].content)
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

    # Session automatically cleaned up when exiting the context manager
    print("✓ Browser session properly closed after all operations")


async def advanced_stateful_example():
    """Advanced example showing complex browser automation with form filling and validation.

    This demonstrates a real-world scenario where session persistence is critical.
    """
    print("\n" + "=" * 80)
    print("ADVANCED EXAMPLE: Complex Browser Automation with Session State")
    print("=" * 80)

    client = MultiServerMCPClient(
        {
            "playwright": {
                "command": "npx",
                "args": [
                    "@playwright/mcp@latest",
                    "--headless",
                ],
                "transport": "stdio",
            }
        }
    )

    async with client.session("playwright") as session:
        tools = await load_mcp_tools(session)

        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model="gpt-4", temperature=0)
        model_with_tools = model.bind_tools(tools)

        memory = MemorySaver()
        agent = create_agent(
            model_with_tools,
            tools,
            debug=False,  # Less verbose for this example
            system_prompt="""You are an expert web automation engineer. 
            Use Playwright tools to interact with web pages and validate functionality.
            Always maintain the browser session state across operations.""",
            checkpointer=memory,
        )

        # Complex test scenario requiring persistent session
        test_scenario = """
        Perform a comprehensive test of a web form:
        1. Navigate to https://httpbin.org/forms/post
        2. Fill in the customer name field with "Test User"
        3. Fill in the telephone field with "555-1234"
        4. Fill in the email field with "test@example.com"
        5. Select "Large" for the pizza size
        6. Check the "Bacon" topping checkbox
        7. Fill in the delivery instructions with "Leave at door"
        8. Take a screenshot of the filled form
        9. Submit the form
        10. Verify the submission was successful by checking the response page
        """

        try:
            config = {"configurable": {"thread_id": "form-test"}}
            response = await agent.ainvoke(
                {"messages": [HumanMessage(content=test_scenario)]},
                config=config,
            )
            print("✓ Complex form automation completed successfully!")
            print("All operations executed in a single browser session")
        except Exception as e:
            print(f"Error during form automation: {e}")


async def main():
    """Run all examples to demonstrate the difference between stateless and stateful MCP usage."""
    print("\n" + "🔍" * 40)
    print("MCP STATEFUL BROWSER AUTOMATION EXAMPLES")
    print("🔍" * 40)
    print("""
This script demonstrates the critical difference between stateless and stateful
MCP tool usage for browser automation. 

Key Learning:
- Stateless (default): Browser closes after each tool call → State lost → Errors
- Stateful (recommended): Browser persists across all tool calls → State maintained → Success

The stateful approach is ESSENTIAL for:
- Browser automation (Playwright, Puppeteer)
- Database connections that need transactions
- Any tools requiring persistent state between calls
    """)

    # Run the problematic stateless example
    try:
        await stateless_example_problematic()
    except Exception as e:
        print(f"Stateless example failed (expected): {e}")

    # Run the correct stateful example
    await stateful_example_correct()

    # Run the advanced example
    await advanced_stateful_example()

    print("\n" + "=" * 80)
    print("SUMMARY: Always use stateful sessions for browser automation!")
    print("Pattern: async with client.session('server_name') as session:")
    print("         tools = await load_mcp_tools(session)")
    print("=" * 80)


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())
