"""Test chat agents in various scenarios."""

from typing import Set

import pytest

from langchain.agents.agent_types import AgentType
from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.llm_math.base import LLMMathChain
from langchain.chat_models.openai import ChatOpenAI
from langchain.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain.tools.plugin import AIPluginTool

TEST_CASES = [
    (
        "What's the current time in NYC?",
        {"DuckDuckGo Search"},
    ),
    ("What is a shoe that's available on Klarna?", {"KlarnaProducts"}),
    ("What's 3*4.2*1.7", {"Calculator"}),
]


@pytest.mark.parametrize("query, used_tools", TEST_CASES)
def test_chat_agent(query: str, used_tools: Set[str]) -> None:
    """Test chat agent."""
    llm = ChatOpenAI(temperature=0)
    llm_math_chain = LLMMathChain(llm=llm)
    tools = [
        DuckDuckGoSearchRun(),
        AIPluginTool.from_plugin_url(
            "https://www.klarna.com/.well-known/ai-plugin.json"
        ),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for doing calculations",
        ),
    ]
    agent_executor = initialize_agent(
        tools,
        llm,
        AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        return_intermediate_steps=True,
    )
    result = agent_executor({"input": query})
    intermediate_steps = result["intermediate_steps"]
    tool_sequences = [act.tool for act, _ in intermediate_steps]
    assert set(tool_sequences) == used_tools
