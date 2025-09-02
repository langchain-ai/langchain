from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

@tool
async def calculator(a: int, b: int) -> str:
    """
    This tool is used to calculate the sum of two numbers.
    """
    return f"{a} + {b} = {a + b}"

agent = create_agent("openai:gpt-5-nano", tools=[calculator], prompt="Please call the calculator tool.")

result = agent.invoke({"messages": [HumanMessage(content="What is 10 + 10?")]})
for msg in result["messages"]:
    msg.pretty_print()
