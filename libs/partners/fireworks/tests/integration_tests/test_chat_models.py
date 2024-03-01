"""Test ChatFireworks API wrapper

You will need FIREWORKS_API_KEY set in your environment to run these tests.
"""

import json

from langchain_core.messages import AIMessage
from langchain_core.pydantic_v1 import BaseModel

from langchain_fireworks import ChatFireworks


def test_chat_fireworks_call() -> None:
    """Test valid call to fireworks."""
    llm = ChatFireworks(
        model="accounts/fireworks/models/firefunction-v1", temperature=0
    )

    resp = llm.invoke("Hello!")
    assert isinstance(resp, AIMessage)

    assert len(resp.content) > 0


def test_tool_choice() -> None:
    """Test that tool choice is respected."""
    llm = ChatFireworks(
        model="accounts/fireworks/models/firefunction-v1", temperature=0
    )

    class MyTool(BaseModel):
        name: str
        age: int

    with_tool = llm.bind_tools([MyTool], tool_choice="MyTool")

    resp = with_tool.invoke("Who was the 27 year old named Erick?")
    assert isinstance(resp, AIMessage)
    assert resp.content == ""  # should just be tool call
    tool_calls = resp.additional_kwargs["tool_calls"]
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call["function"]["name"] == "MyTool"
    assert json.loads(tool_call["function"]["arguments"]) == {
        "age": 27,
        "name": "Erick",
    }
    assert tool_call["type"] == "function"


def test_tool_choice_bool() -> None:
    """Test that tool choice is respected just passing in True."""

    llm = ChatFireworks(
        model="accounts/fireworks/models/firefunction-v1", temperature=0
    )

    class MyTool(BaseModel):
        name: str
        age: int

    with_tool = llm.bind_tools([MyTool], tool_choice=True)

    resp = with_tool.invoke("Who was the 27 year old named Erick?")
    assert isinstance(resp, AIMessage)
    assert resp.content == ""  # should just be tool call
    tool_calls = resp.additional_kwargs["tool_calls"]
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call["function"]["name"] == "MyTool"
    assert json.loads(tool_call["function"]["arguments"]) == {
        "age": 27,
        "name": "Erick",
    }
    assert tool_call["type"] == "function"
