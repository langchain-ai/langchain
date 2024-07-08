"""Test formatting functionality."""

from typing import Union

import pytest
from langchain_core.agents import AgentAction, AgentActionMessageLog, AgentFinish
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, Generation
from langchain_core.prompt_values import ChatPromptValueConcrete, StringPromptValue
from langchain_core.pydantic_v1 import BaseModel, ValidationError


def test_serialization_of_wellknown_objects() -> None:
    """Test that pydantic is able to serialize and deserialize well known objects."""

    class WellKnownLCObject(BaseModel):
        """A well known LangChain object."""

        __root__: Union[
            Document,
            HumanMessage,
            SystemMessage,
            ChatMessage,
            FunctionMessage,
            AIMessage,
            HumanMessageChunk,
            SystemMessageChunk,
            ChatMessageChunk,
            FunctionMessageChunk,
            AIMessageChunk,
            StringPromptValue,
            ChatPromptValueConcrete,
            AgentFinish,
            AgentAction,
            AgentActionMessageLog,
            ChatGeneration,
            Generation,
            ChatGenerationChunk,
        ]

    lc_objects = [
        HumanMessage(content="human"),
        HumanMessageChunk(content="human"),
        AIMessage(content="ai"),
        AIMessageChunk(content="ai"),
        SystemMessage(content="sys"),
        SystemMessageChunk(content="sys"),
        FunctionMessage(
            name="func",
            content="func",
        ),
        FunctionMessageChunk(
            name="func",
            content="func",
        ),
        ChatMessage(
            role="human",
            content="human",
        ),
        ChatMessageChunk(
            role="human",
            content="human",
        ),
        StringPromptValue(text="hello"),
        ChatPromptValueConcrete(messages=[HumanMessage(content="human")]),
        Document(page_content="hello"),
        AgentFinish(return_values={}, log=""),
        AgentAction(tool="tool", tool_input="input", log=""),
        AgentActionMessageLog(
            tool="tool",
            tool_input="input",
            log="",
            message_log=[HumanMessage(content="human")],
        ),
        Generation(
            text="hello",
            generation_info={"info": "info"},
        ),
        ChatGeneration(
            message=HumanMessage(content="human"),
        ),
        ChatGenerationChunk(
            message=HumanMessageChunk(content="cat"),
        ),
    ]

    for lc_object in lc_objects:
        d = lc_object.dict()
        assert "type" in d, f"Missing key `type` for {type(lc_object)}"
        obj1 = WellKnownLCObject.parse_obj(d)
        assert type(obj1.__root__) is type(lc_object), f"failed for {type(lc_object)}"

    with pytest.raises(ValidationError):
        # Make sure that specifically validation error is raised
        WellKnownLCObject.parse_obj({})
