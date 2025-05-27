# Create server parameters for stdio connection
import asyncio

from dotenv import find_dotenv, load_dotenv
from langchain_gigachat import GigaChat
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from rich import print as rprint

load_dotenv(find_dotenv())

# LLM GigaChat
model = GigaChat(model="GigaChat-2-Max",
                verify_ssl_certs=False,
                streaming=False,
                max_tokens=8000,
                timeout=600)

def _log(ans):
    for message in ans['messages']:
        rprint(f"[{type(message).__name__}] {message.content} {getattr(message, 'tool_calls', '')}")


async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["math_server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)

            # Create and run the agent
            agent = create_react_agent(model, tools)

            agent_response = await agent.ainvoke({"messages": [
                {"role": "user", "content": "Сколько будет (3 + 5) x 12?"}]})
            _log(agent_response)
            
            agent_response = await agent.ainvoke({"messages": [
                {"role": "user", "content": "Найди сколько лет Джону Доу?"}]})
            _log(agent_response)

# Run the main function
asyncio.run(main())