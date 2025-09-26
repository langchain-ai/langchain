"""Additional tests for RunnableConfig propagation into tools.

Covers tools whose _run/_arun signatures annotate the config parameter as
Optional[RunnableConfig] and Union[RunnableConfig, None], ensuring sync and async
agent paths propagate config with and without explicit config provided.
"""

from __future__ import annotations

from typing import Any, ClassVar, Optional, Union, cast

import pytest
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool

from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent

PROMPT_TMPL = """Answer the following questions as best you can. You have access to the
    following tools:

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


class OptionalConfigTool(BaseTool):
    """Tool capturing Optional[RunnableConfig] in _run/_arun."""

    name: str = "optional_config_tool"
    description: str = "captures optional config"

    # class-level capture for assertions
    last_sync_config: ClassVar[Optional[RunnableConfig]] = None
    last_async_config: ClassVar[Optional[RunnableConfig]] = None

    def _run(
        self, _: str, config: Optional[RunnableConfig] = None, **__: object
    ) -> str:
        OptionalConfigTool.last_sync_config = config
        return "ok"

    async def _arun(
        self, _: str, config: Optional[RunnableConfig] = None, **__: object
    ) -> str:
        OptionalConfigTool.last_async_config = config
        return "ok"


class UnionConfigTool(BaseTool):
    """Tool capturing Union[RunnableConfig, None] in _run/_arun."""

    name: str = "union_config_tool"
    description: str = "captures union config"

    last_sync_config: ClassVar[Optional[RunnableConfig]] = None
    last_async_config: ClassVar[Optional[RunnableConfig]] = None

    def _run(
        self, _: str, config: Union[RunnableConfig, None] = None, **__: object
    ) -> str:
        UnionConfigTool.last_sync_config = config
        return "ok"

    async def _arun(
        self, _: str, config: Union[RunnableConfig, None] = None, **__: object
    ) -> str:
        UnionConfigTool.last_async_config = config
        return "ok"


def _make_agent_and_executor(tool: BaseTool) -> AgentExecutor:
    llm = FakeListLLM(
        responses=[
            f"I should use the {tool.name}.\nAction: {tool.name}\nAction Input: test",
            "Final Answer: done",
        ]
    )
    prompt = PromptTemplate.from_template(PROMPT_TMPL)
    agent = create_react_agent(llm, [tool], prompt)
    return AgentExecutor(agent=agent, tools=[tool], verbose=False)


def test_optional_config_sync_with_and_without_config() -> None:
    tool = OptionalConfigTool()
    exec_ = _make_agent_and_executor(tool)

    # With explicit config
    test_config: RunnableConfig = {
        "configurable": {"k": "v"},
        "metadata": {"m": 1},
        "tags": ["t"],
    }
    OptionalConfigTool.last_sync_config = None
    exec_.invoke({"input": "go"}, config=test_config)
    # Ensure content from provided config is visible
    cfg = cast("Any", OptionalConfigTool.last_sync_config)
    assert cfg.get("configurable", {}).get("k") == "v"

    # Without explicit config
    OptionalConfigTool.last_sync_config = None
    exec_.invoke({"input": "go"})
    # BaseTool ensures a child config is still constructed
    cfg2 = cast("Any", OptionalConfigTool.last_sync_config)
    assert isinstance(cfg2, dict)


def test_union_config_sync_with_and_without_config() -> None:
    tool = UnionConfigTool()
    exec_ = _make_agent_and_executor(tool)

    # With explicit config
    test_config: RunnableConfig = {"configurable": {"alpha": True}}
    UnionConfigTool.last_sync_config = None
    exec_.invoke({"input": "go"}, config=test_config)
    cfg3 = cast("Any", UnionConfigTool.last_sync_config)
    assert cfg3.get("configurable", {}).get("alpha") is True

    # Without explicit config
    UnionConfigTool.last_sync_config = None
    exec_.invoke({"input": "go"})
    cfg4 = cast("Any", UnionConfigTool.last_sync_config)
    assert isinstance(cfg4, dict)


@pytest.mark.asyncio
async def test_optional_config_async_with_and_without_config() -> None:
    tool = OptionalConfigTool()
    exec_ = _make_agent_and_executor(tool)

    # With explicit config
    test_config: RunnableConfig = {"metadata": {"src": "unit"}}
    OptionalConfigTool.last_async_config = None
    await exec_.ainvoke({"input": "go"}, config=test_config)
    cfg5 = cast("Any", OptionalConfigTool.last_async_config)
    assert cfg5.get("metadata", {}).get("src") == "unit"

    # Without explicit config
    OptionalConfigTool.last_async_config = None
    await exec_.ainvoke({"input": "go"})
    cfg6 = cast("Any", OptionalConfigTool.last_async_config)
    assert isinstance(cfg6, dict)


@pytest.mark.asyncio
async def test_union_config_async_with_and_without_config() -> None:
    tool = UnionConfigTool()
    exec_ = _make_agent_and_executor(tool)

    # With explicit config
    test_config: RunnableConfig = {"metadata": {"mode": "async"}}
    UnionConfigTool.last_async_config = None
    await exec_.ainvoke({"input": "go"}, config=test_config)
    cfg7 = cast("Any", UnionConfigTool.last_async_config)
    assert cfg7.get("metadata", {}).get("mode") == "async"

    # Without explicit config
    UnionConfigTool.last_async_config = None
    await exec_.ainvoke({"input": "go"})
    cfg8 = cast("Any", UnionConfigTool.last_async_config)
    assert isinstance(cfg8, dict)
