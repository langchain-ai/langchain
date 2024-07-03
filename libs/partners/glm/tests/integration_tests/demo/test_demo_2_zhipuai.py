import logging
import logging.config

import zhipuai
from langchain_core.messages import AIMessage
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough,
)
from langchain_core.tools import tool
from zhipuai import ZhipuAI

from langchain_glm import ChatZhipuAI

logger = logging.getLogger(__name__)


def test_openai_demo_2_tools(logging_conf):
    logging.config.dictConfig(logging_conf)  # type: ignore
    client = ZhipuAI()
    response = client.chat.completions.create(
        model="glm-4-0520",
        messages=[{"role": "user", "content": "帮我查询天气"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ],
        top_p=0.7,
        temperature=0.1,
        max_tokens=2000,
    )
    logger.info("\033[1;32m" + f"client: {response}" + "\033[0m")


@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int


@tool
def add(first_int: int, second_int: int) -> int:
    "Add two integers."
    return first_int + second_int


@tool
def exponentiate(base: int, exponent: int) -> int:
    "Exponentiate the base to the exponent power."
    return base**exponent


def test_openai_demo1_tool_use():
    llm = ChatZhipuAI(
        model="glm-4-0520",
        streaming=True,
    )

    tools = [multiply, exponentiate, add]
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {tool.name: tool for tool in tools}

    def call_tools(msg: AIMessage) -> Runnable:
        """Simple sequential tool calling helper."""
        tool_map = {tool.name: tool for tool in tools}
        tool_calls = msg.tool_calls.copy()
        for tool_call in tool_calls:
            tool_call["output"] = tool_map[tool_call["name"]].invoke(tool_call["args"])
        return tool_calls

    chain = llm_with_tools | call_tools
    out = chain.invoke(
        "What's 23 times 7, and what's five times 18 and add a million plus a billion and cube thirty-seven"
    )
    print(out)
