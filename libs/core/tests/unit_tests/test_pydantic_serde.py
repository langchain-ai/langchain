"""A set of tests that verifies that Union discrimination works correctly with
the various pydantic base models.

These tests can uncover issues that will also arise during regular instantiation
of the models (i.e., not necessarily from loading or dumping JSON).
"""

import pytest
from pydantic import RootModel, ValidationError

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
from langchain_core.prompt_values import ChatPromptValueConcrete


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

    Model = RootModel[AnyMessage]

    for lc_object in lc_objects:
        d = lc_object.model_dump()
        assert "type" in d, f"Missing key `type` for {type(lc_object)}"
        obj1 = Model.model_validate(d)
        assert type(obj1.root) is type(lc_object), f"failed for {type(lc_object)}"

    with pytest.raises((TypeError, ValidationError)):
        # Make sure that specifically validation error is raised
        Model.model_validate({})


def test_serde_chat_prompt_value():
    prompt = ChatPromptValueConcrete(
        messages=[
            AIMessage(
                content="Hello",
            ),
            HumanMessage(
                content=" World",
            )
        ]
    )

    # Derived = RootModel[Sequence[Any_]]


def test_kookoo():
    import pydantic
    from pydantic import __version__

    from typing import Annotated, Union, Literal, Sequence, Any
    from pydantic import BaseModel, Field, Tag, RootModel, Discriminator
    import pprint
    class Base(BaseModel):
        y: int = 'hello'
        type: Literal['base'] = 'base'

    class Foo(Base):
        type: Literal['foo'] = 'foo'
        x: int

    class Bar(Base):
        type: Literal['bar'] = 'bar'
        x: int

    FooOrBar = Annotated[Union[Foo, Bar], Field(discriminator="type")]


    class BaseContainer(BaseModel):
        messages: Sequence[Base]

    class Container(BaseModel):
        messages: Sequence[FooOrBar]


    Container(messages=[Foo(x=5), Bar(x=2), Foo(x=10)])
