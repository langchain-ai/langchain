"""Test formatting functionality."""

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
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, Generation
from langchain_core.prompt_values import ChatPromptValueConcrete, StringPromptValue
from pydantic import RootModel, ValidationError


@pytest.mark.xfail(reason="TODO: FIX BEFORE 0.3 RELEASE")
def test_serialization_of_wellknown_objects() -> None:
    """Test that pydantic is able to serialize and deserialize well known objects."""
    well_known_lc_object = RootModel[
        Document
        | HumanMessage
        | SystemMessage
        | ChatMessage
        | FunctionMessage
        | FunctionMessageChunk
        | AIMessage
        | HumanMessageChunk
        | SystemMessageChunk
        | ChatMessageChunk
        | AIMessageChunk
        | StringPromptValue
        | ChatPromptValueConcrete
        | AgentFinish
        | AgentAction
        | AgentActionMessageLog
        | ChatGeneration
        | Generation
        | ChatGenerationChunk,
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
        ChatPromptValueConcrete(messages=[AIMessage(content="foo")]),
        ChatPromptValueConcrete(messages=[HumanMessage(content="human")]),
        ChatPromptValueConcrete(
            messages=[ToolMessage(content="foo", tool_call_id="bar")],
        ),
        ChatPromptValueConcrete(messages=[SystemMessage(content="foo")]),
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
        d = lc_object.model_dump()
        assert "type" in d, f"Missing key `type` for {type(lc_object)}"
        obj1 = well_known_lc_object.model_validate(d)
        assert type(obj1.root) is type(lc_object), f"failed for {type(lc_object)}"

    with pytest.raises((TypeError, ValidationError)):
        # Make sure that specifically validation error is raised
        well_known_lc_object.model_validate({})
