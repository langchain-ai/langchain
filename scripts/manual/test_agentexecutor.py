#!/usr/bin/env python3
"""Manual smoke test for AgentExecutor.invoke."""

from langchain.agents import AgentExecutor
from langchain_core.language_models.fake import FakeListLLM
from langchain.agents.react.agent import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool


class DummyTool(BaseTool):
    name: str = "dummy"
    description: str = "dummy tool"

    def _run(self, query: str, **kwargs) -> str:
        return "dummy result"


def main() -> None:
    tool = DummyTool()
    llm = FakeListLLM(responses=["Final Answer: test"])
    template = """Use this tool: {tools}
Tool names: {tool_names}
Question: {input}
{agent_scratchpad}"""
    prompt = PromptTemplate.from_template(template)
    agent = create_react_agent(llm, [tool], prompt)
    executor = AgentExecutor(agent=agent, tools=[tool])
    result = executor.invoke({"input": "test"})
    print(result)


if __name__ == "__main__":
    main()
