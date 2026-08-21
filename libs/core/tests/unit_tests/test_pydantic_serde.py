"""Test pydantic SerDe.

A set of tests that verifies that Union discrimination works correctly with
the various pydantic base models.

These tests can uncover issues that will also arise during regular instantiation
of the models (i.e., not necessarily from loading or dumping JSON).
"""

import pytest
from pydantic import RootModel, ValidationError

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
)


def test_serde_any_message() -> None:
    """Test AnyMessage() serder."""
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
    ]

    model = RootModel[AnyMessage]

    for lc_object in lc_objects:
        d = lc_object.model_dump()
        assert "type" in d, f"Missing key `type` for {type(lc_object)}"
        obj1 = model.model_validate(d)
        assert type(obj1.root) is type(lc_object), f"failed for {type(lc_object)}"

    with pytest.raises((TypeError, ValidationError)):
        # Make sure that specifically validation error is raised
        model.model_validate({})


def test_serde_document_first_union_does_not_raise_type_error() -> None:
    """Test that union validation can continue past `Document`."""
    model = RootModel[Document | HumanMessage]

    human_payload = HumanMessage(content="human").model_dump()
    parsed = model.model_validate(human_payload)

    assert type(parsed.root) is HumanMessage


def test_document_missing_page_content_raises_validation_error() -> None:
    """Test that missing `page_content` uses pydantic validation errors."""
    model = RootModel[Document | HumanMessage]

    with pytest.raises(ValidationError):
        model.model_validate({"type": "Document"})


def test_serde_agent_union_does_not_raise_type_error() -> None:
    """Test that union validation can continue past agent schema types."""
    model = RootModel[AgentFinish | AgentAction | HumanMessage]

    human_payload = HumanMessage(content="human").model_dump()
    parsed = model.model_validate(human_payload)

    assert type(parsed.root) is HumanMessage
