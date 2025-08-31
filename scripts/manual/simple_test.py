#!/usr/bin/env python3
"""Manual test to verify RunnableConfig is passed to tools."""

from typing import Optional
from langchain_core.tools import BaseTool
from langchain_core.runnables.config import RunnableConfig
from langchain.agents import AgentExecutor
from langchain_core.language_models.fake import FakeListLLM
from langchain.agents.react.agent import create_react_agent
from langchain_core.prompts import PromptTemplate


class TestTool(BaseTool):
    name: str = "test_tool"
    description: str = "Test tool that checks config"
    last_config: Optional[RunnableConfig] = None

    def _run(self, query: str, config: Optional[RunnableConfig] = None, **kwargs) -> str:
        TestTool.last_config = config
        return f"Tool executed, config received: {config is not None}"


def main() -> None:
    TestTool.last_config = None

    tool = TestTool()
    llm = FakeListLLM(
        responses=[
            "Action: test_tool\nAction Input: test",
            "Final Answer: done",
        ]
    )

    template = """Use this tool: {tools}
Tool names: {tool_names}
Question: {input}
{agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template)
    agent = create_react_agent(llm, [tool], prompt)
    executor = AgentExecutor(agent=agent, tools=[tool])
    config = {"metadata": {"test": "value"}}
    result = executor.invoke({"input": "test"}, config=config)
    print(result)


if __name__ == "__main__":
    main()
