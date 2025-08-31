#!/usr/bin/env python3
"""Manual validation that RunnableConfig is passed to tools in AgentExecutor."""

import asyncio
from typing import Optional, Any

from langchain_core.tools import BaseTool
from langchain_core.runnables.config import RunnableConfig
from langchain.agents import AgentExecutor
from langchain_core.language_models.fake import FakeListLLM
from langchain.agents.react.agent import create_react_agent
from langchain_core.prompts import PromptTemplate


class ConfigCaptureTool(BaseTool):
    name: str = "config_capture"
    description: str = "Captures the RunnableConfig passed to tools"
    captured_config: Optional[RunnableConfig] = None

    def _run(self, query: str, run_manager=None, config: Optional[RunnableConfig] = None, **kwargs: Any) -> str:
        self.captured_config = config
        return f"Config captured. Config present: {config is not None}"

    async def _arun(self, query: str, run_manager=None, config: Optional[RunnableConfig] = None, **kwargs: Any) -> str:
        self.captured_config = config
        return f"Config captured async. Config present: {config is not None}"


def _make_executor(tool: BaseTool) -> AgentExecutor:
    fake_llm = FakeListLLM(
        responses=[
            "Action: config_capture\nAction Input: test",
            "Final Answer: Tool executed successfully",
        ]
    )
    template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
    prompt = PromptTemplate.from_template(template)
    agent = create_react_agent(fake_llm, [tool], prompt)
    return AgentExecutor(agent=agent, tools=[tool], verbose=False)


def main() -> None:
    tool = ConfigCaptureTool()
    exec_ = _make_executor(tool)
    test_config = {"configurable": {"test_key": "test_value"}, "metadata": {"source": "test"}, "tags": ["test_tag"]}
    result = exec_.invoke({"input": "Test the config"}, config=test_config)
    print(result)

    tool.captured_config = None

    async def _run_async() -> None:
        result_async = await exec_.ainvoke({"input": "Test the config async"}, config=test_config)
        print(result_async)

    asyncio.run(_run_async())


if __name__ == "__main__":
    main()
