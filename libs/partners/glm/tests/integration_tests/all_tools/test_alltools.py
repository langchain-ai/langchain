import logging
import logging.config

import pytest
from langchain.agents import tool
from langchain.tools.shell import ShellTool
from pydantic.v1 import BaseModel, Extra, Field

from langchain_glm.agent_toolkits import BaseToolOutput
from langchain_glm.agents.zhipuai_all_tools import (
    ZhipuAIAllToolsRunnable,
)
from langchain_glm.agents.zhipuai_all_tools.base import (
    AllToolsAction,
    AllToolsActionToolEnd,
    AllToolsActionToolStart,
    AllToolsFinish,
    AllToolsLLMStatus,
)
from langchain_glm.callbacks.agent_callback_handler import (
    AgentStatus,
)


@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int


@tool
def add(first_int: int, second_int: int) -> int:
    "Add two integers."
    return first_int + second_int


@tool
def exp(exponent_num: int, base: int) -> int:
    "Exponentiate the base to the exponent power."
    return base**exponent_num


@pytest.mark.asyncio
async def test_all_tools_func(logging_conf):
    logging.config.dictConfig(logging_conf)  # type: ignore

    agent_executor = ZhipuAIAllToolsRunnable.create_agent_executor(
        model_name="glm-4-alltools",
        tools=[multiply, exp, add],
    )
    chat_iterator = agent_executor.invoke(chat_input="计算下 2 乘以 5")
    async for item in chat_iterator:
        if isinstance(item, AllToolsAction):
            print("AllToolsAction:" + str(item.to_json()))

        elif isinstance(item, AllToolsFinish):
            print("AllToolsFinish:" + str(item.to_json()))

        elif isinstance(item, AllToolsActionToolStart):
            print("AllToolsActionToolStart:" + str(item.to_json()))

        elif isinstance(item, AllToolsActionToolEnd):
            print("AllToolsActionToolEnd:" + str(item.to_json()))
        elif isinstance(item, AllToolsLLMStatus):
            if item.status == AgentStatus.llm_end:
                print("llm_end:" + item.text)


@pytest.mark.asyncio
async def test_all_tools_code_interpreter(logging_conf):
    logging.config.dictConfig(logging_conf)  # type: ignore

    agent_executor = ZhipuAIAllToolsRunnable.create_agent_executor(
        model_name="glm-4-alltools",
        tools=[
            {"type": "code_interpreter"},
        ],
    )
    chat_iterator = agent_executor.invoke(chat_input="计算下 2 乘以 5")
    async for item in chat_iterator:
        if isinstance(item, AllToolsAction):
            print("AllToolsAction:" + str(item.to_json()))

        elif isinstance(item, AllToolsFinish):
            print("AllToolsFinish:" + str(item.to_json()))

        elif isinstance(item, AllToolsActionToolStart):
            print("AllToolsActionToolStart:" + str(item.to_json()))

        elif isinstance(item, AllToolsActionToolEnd):
            print("AllToolsActionToolEnd:" + str(item.to_json()))
        elif isinstance(item, AllToolsLLMStatus):
            if item.status == AgentStatus.llm_end:
                print("llm_end:" + item.text)


@pytest.mark.asyncio
async def test_all_tools_code_interpreter_sandbox_none(logging_conf):
    logging.config.dictConfig(logging_conf)  # type: ignore

    agent_executor = ZhipuAIAllToolsRunnable.create_agent_executor(
        model_name="glm-4-alltools",
        tools=[{"type": "code_interpreter", "code_interpreter": {"sandbox": "none"}}],
    )
    chat_iterator = agent_executor.invoke(
        chat_input="看下本地文件有哪些，告诉我你用的是什么文件,查看当前目录"
    )
    async for item in chat_iterator:
        if isinstance(item, AllToolsAction):
            print("AllToolsAction:" + str(item.to_json()))

        elif isinstance(item, AllToolsFinish):
            print("AllToolsFinish:" + str(item.to_json()))

        elif isinstance(item, AllToolsActionToolStart):
            print("AllToolsActionToolStart:" + str(item.to_json()))

        elif isinstance(item, AllToolsActionToolEnd):
            print("AllToolsActionToolEnd:" + str(item.to_json()))
        elif isinstance(item, AllToolsLLMStatus):
            if item.status == AgentStatus.llm_end:
                print("llm_end:" + item.text)

    chat_iterator = agent_executor.invoke(chat_input="打印下test_alltools.py")
    async for item in chat_iterator:
        if isinstance(item, AllToolsAction):
            print("AllToolsAction:" + str(item.to_json()))

        elif isinstance(item, AllToolsFinish):
            print("AllToolsFinish:" + str(item.to_json()))

        elif isinstance(item, AllToolsActionToolStart):
            print("AllToolsActionToolStart:" + str(item.to_json()))

        elif isinstance(item, AllToolsActionToolEnd):
            print("AllToolsActionToolEnd:" + str(item.to_json()))
        elif isinstance(item, AllToolsLLMStatus):
            if item.status == AgentStatus.llm_end:
                print("llm_end:" + item.text)


@pytest.mark.asyncio
async def test_all_tools_drawing_tool(logging_conf):
    logging.config.dictConfig(logging_conf)  # type: ignore

    agent_executor = ZhipuAIAllToolsRunnable.create_agent_executor(
        model_name="glm-4-alltools",
        tools=[{"type": "drawing_tool"}],
    )
    chat_iterator = agent_executor.invoke(chat_input="给我画一张猫咪的图片，要是波斯猫")
    async for item in chat_iterator:
        if isinstance(item, AllToolsAction):
            print("AllToolsAction:" + str(item.to_json()))

        elif isinstance(item, AllToolsFinish):
            print("AllToolsFinish:" + str(item.to_json()))

        elif isinstance(item, AllToolsActionToolStart):
            print("AllToolsActionToolStart:" + str(item.to_json()))

        elif isinstance(item, AllToolsActionToolEnd):
            print("AllToolsActionToolEnd:" + str(item.to_json()))
        elif isinstance(item, AllToolsLLMStatus):
            if item.status == AgentStatus.llm_end:
                print("llm_end:" + item.text)


@pytest.mark.asyncio
async def test_all_tools_web_browser(logging_conf):
    logging.config.dictConfig(logging_conf)  # type: ignore

    agent_executor = ZhipuAIAllToolsRunnable.create_agent_executor(
        model_name="glm-4-alltools",
        tools=[{"type": "web_browser"}],
    )
    chat_iterator = agent_executor.invoke(chat_input="帮我搜索今天的新闻")
    async for item in chat_iterator:
        if isinstance(item, AllToolsAction):
            print("AllToolsAction:" + str(item.to_json()))

        elif isinstance(item, AllToolsFinish):
            print("AllToolsFinish:" + str(item.to_json()))

        elif isinstance(item, AllToolsActionToolStart):
            print("AllToolsActionToolStart:" + str(item.to_json()))

        elif isinstance(item, AllToolsActionToolEnd):
            print("AllToolsActionToolEnd:" + str(item.to_json()))
        elif isinstance(item, AllToolsLLMStatus):
            if item.status == AgentStatus.llm_end:
                print("llm_end:" + item.text)


@pytest.mark.asyncio
async def test_all_tools_start(logging_conf):
    logging.config.dictConfig(logging_conf)  # type: ignore

    agent_executor = ZhipuAIAllToolsRunnable.create_agent_executor(
        model_name="glm-4-alltools",
        tools=[
            {"type": "code_interpreter", "code_interpreter": {"sandbox": "none"}},
            {"type": "web_browser"},
            {"type": "drawing_tool"},
        ],
    )
    chat_iterator = agent_executor.invoke(
        chat_input="帮我查询2018年至2024年，每年五一假期全国旅游出行数据，并绘制成柱状图展示数据趋势。"
    )

    async for item in chat_iterator:
        if isinstance(item, AllToolsAction):
            print("AllToolsAction:" + str(item.to_json()))

        elif isinstance(item, AllToolsFinish):
            print("AllToolsFinish:" + str(item.to_json()))

        elif isinstance(item, AllToolsActionToolStart):
            print("AllToolsActionToolStart:" + str(item.to_json()))

        elif isinstance(item, AllToolsActionToolEnd):
            print("AllToolsActionToolEnd:" + str(item.to_json()))
        elif isinstance(item, AllToolsLLMStatus):
            if item.status == AgentStatus.llm_end:
                print("llm_end:" + item.text)
