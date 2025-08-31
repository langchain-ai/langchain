#!/usr/bin/env python3
"""Manual validation for RunnableConfig propagation (not part of unit tests)."""

import asyncio
from typing import Optional

from langchain_core.tools import BaseTool
from langchain_core.runnables.config import RunnableConfig
from langchain.agents import AgentExecutor
from langchain_core.language_models.fake import FakeListLLM
from langchain.agents.react.agent import create_react_agent
from langchain_core.prompts import PromptTemplate


class ValidationTool(BaseTool):
    name: str = "validation_tool"
    description: str = "A tool for validating RunnableConfig propagation"

    def _run(self, query: str, config: Optional[RunnableConfig] = None, **kwargs) -> str:
        if config is not None:
            metadata = config.get("metadata", {})
            configurable = config.get("configurable", {})
            return f"SUCCESS: Config received with metadata={metadata}, configurable={configurable}"
        return "FAILED: No config received"

    async def _arun(self, query: str, config: Optional[RunnableConfig] = None, **kwargs) -> str:
        return self._run(query, config, **kwargs)


def _make_executor(tool: BaseTool) -> AgentExecutor:
    llm = FakeListLLM(
        responses=[
            "Action: validation_tool\nAction Input: test",
            "Final Answer: validation complete",
        ]
    )
    template = """Use this tool: {tools}
Tool names: {tool_names}
Question: {input}
{agent_scratchpad}"""
    prompt = PromptTemplate.from_template(template)
    agent = create_react_agent(llm, [tool], prompt)
    return AgentExecutor(agent=agent, tools=[tool])


def run_sync() -> None:
    tool = ValidationTool()
    executor = _make_executor(tool)
    config = {"metadata": {"test_run": "sync"}, "configurable": {"setting": "value"}}
    result = executor.invoke({"input": "validate config"}, config=config)
    print(result)


async def run_async() -> None:
    tool = ValidationTool()
    executor = _make_executor(tool)
    config = {"metadata": {"test_run": "async"}, "configurable": {"setting": "async_value"}}
    result = await executor.ainvoke({"input": "validate config async"}, config=config)
    print(result)


if __name__ == "__main__":
    run_sync()
    asyncio.run(run_async())
