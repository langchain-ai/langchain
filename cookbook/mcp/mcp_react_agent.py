"""MCP ReAct агент с поддержкой памяти."""
import asyncio
import json

from dotenv import load_dotenv, find_dotenv
from langchain_gigachat import GigaChat
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from rich import print

# Load environment variables
load_dotenv(find_dotenv())

# Initialize GigaChat with optimized settings
GIGA_CHAT = GigaChat(
    model="GigaChat-2-Max",
    verify_ssl_certs=False,
    streaming=False,
    max_tokens=8000,
)

# Load MCP configuration
with open("mcp_config.json", "r", encoding="utf-8") as f:
    MCP_CONFIG = json.load(f)


async def run_interactive_session(agent):
    """Run interactive chat session with the agent."""
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit", ""]:
            break

        response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": user_input}]}, 
            config={"configurable": {"thread_id": "1"}}
        )
        
        # Log tool calls
        messages = response['messages']
        for msg in messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    print(f"🔧 Tool: {tool_call['name']} | Args: {tool_call['args']}")
        
        print(f"Agent: {response['messages'][-1].content}")


async def main():
    """Main entry point for the MCP React agent."""
    async with MultiServerMCPClient(MCP_CONFIG) as mcp_client:
        # Create agent with MCP tools and memory
        agent = create_react_agent(
            GIGA_CHAT,
            tools=mcp_client.get_tools(),
            prompt="Ты полезный ассистент.",
            checkpointer=MemorySaver()
        )
        
        await run_interactive_session(agent)


if __name__ == "__main__":
    asyncio.run(main())
    
