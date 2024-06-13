"""Test DeepInfra API wrapper."""
from typing import List

from langchain_community.llms.deepinfra import DeepInfra

from langchain_core.messages.tool import ToolMessage

from langchain_core.messages.ai import AIMessageChunk, AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field

from libs.core.langchain_core.messages.human import HumanMessage


class GenerateMovieName(BaseModel):
    "Get a movie name from a description"
    description: str


model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
def test_deepinfra_call() -> None:
    """Test valid call to DeepInfra."""
    llm = DeepInfra(model_id=model_id)
    output = llm.invoke("What is 2 + 2?")
    assert isinstance(output, str)


async def test_deepinfra_acall() -> None:
    llm = DeepInfra(model_id=model_id)
    output = await llm.ainvoke("What is 2 + 2?")
    assert llm._llm_type == "deepinfra"
    assert isinstance(output, str)


def test_deepinfra_stream() -> None:
    llm = DeepInfra(model_id=model_id)
    num_chunks = 0
    for chunk in llm.stream("[INST] Hello [/INST] "):
        num_chunks += 1
    assert num_chunks > 0


async def test_deepinfra_astream() -> None:
    llm = DeepInfra(model_id=model_id)
    num_chunks = 0
    async for chunk in llm.astream("[INST] Hello [/INST] "):
        num_chunks += 1
    assert num_chunks > 0


def test_tool_use() -> None:
    llm = DeepInfra(model_id=model_id, temperature = 0)
    llm_with_tool = llm.bind_tools(tools=[GenerateMovieName], tool_choice=True)
    msgs: List = [HumanMessage("It should be a movie explaining humanity in 2133.")]
    ai_msg = llm_with_tool.invoke(msgs)

    assert isinstance(ai_msg, AIMessage)
    assert isinstance(ai_msg.tool_calls, list)
    assert len(ai_msg.tool_calls) == 1
    tool_call = ai_msg.tool_calls[0]
    assert "args" in tool_call

    tool_msg = ToolMessage(
        "Year 2133", tool_call_id=ai_msg.additional_kwargs["tool_calls"][0]["id"]
    )
    msgs.extend([ai_msg, tool_msg])
    llm_with_tool.invoke(msgs)
