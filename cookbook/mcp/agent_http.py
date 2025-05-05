# Create server parameters for stdio connection
import asyncio
import os

from dotenv import find_dotenv, load_dotenv
from langchain_gigachat.chat_models.gigachat import GigaChat
from langgraph.prebuilt import create_react_agent
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.client import MultiServerMCPClient
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
    async with MultiServerMCPClient(
        {
            "math": {
                "url": "http://localhost:8000/sse",
                "transport": "sse",
            }
        }
    ) as client:
        agent = create_react_agent(model, client.get_tools())
        
        agent_response = await agent.ainvoke({"messages": [
            {"role": "user", "content": "Сколько будет (3 + 5) x 12?"}]})
        _log(agent_response)
        
        agent_response = await agent.ainvoke({"messages": [
            {"role": "user", "content": "Найди сколько лет Джону Доу?"}]})
        _log(agent_response)

# Run the main function
asyncio.run(main())