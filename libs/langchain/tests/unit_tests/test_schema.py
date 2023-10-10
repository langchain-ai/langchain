"""Test formatting functionality."""

import unittest
from typing import Union

from langchain.pydantic_v1 import BaseModel
from langchain.schema.messages import (
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
    get_buffer_string,
    messages_from_dict,
    messages_to_dict,
)


class TestGetBufferString(unittest.TestCase):
    def setUp(self) -> None:
        self.human_msg = HumanMessage(content="human")
        self.ai_msg = AIMessage(content="ai")
        self.sys_msg = SystemMessage(content="sys")

    def test_empty_input(self) -> None:
        self.assertEqual(get_buffer_string([]), "")

    def test_valid_single_message(self) -> None:
        expected_output = f"Human: {self.human_msg.content}"
        self.assertEqual(
            get_buffer_string([self.human_msg]),
            expected_output,
        )

    def test_custom_human_prefix(self) -> None:
        prefix = "H"
        expected_output = f"{prefix}: {self.human_msg.content}"
        self.assertEqual(
            get_buffer_string([self.human_msg], human_prefix="H"),
            expected_output,
        )

    def test_custom_ai_prefix(self) -> None:
        prefix = "A"
        expected_output = f"{prefix}: {self.ai_msg.content}"
        self.assertEqual(
            get_buffer_string([self.ai_msg], ai_prefix="A"),
            expected_output,
        )

    def test_multiple_msg(self) -> None:
        msgs = [self.human_msg, self.ai_msg, self.sys_msg]
        expected_output = "\n".join(
            [
                f"Human: {self.human_msg.content}",
                f"AI: {self.ai_msg.content}",
                f"System: {self.sys_msg.content}",
            ]
        )
        self.assertEqual(
            get_buffer_string(msgs),
            expected_output,
        )


def test_multiple_msg() -> None:
    human_msg = HumanMessage(content="human", additional_kwargs={"key": "value"})
    ai_msg = AIMessage(content="ai")
    sys_msg = SystemMessage(content="sys")

    msgs = [
        human_msg,
        ai_msg,
        sys_msg,
    ]
    assert messages_from_dict(messages_to_dict(msgs)) == msgs


def test_distinguish_messages() -> None:
    """Test that pydantic is able to discriminate between similar looking messages."""

    class WellKnownTypes(BaseModel):
        __root__: Union[
            HumanMessage,
            AIMessage,
            SystemMessage,
            FunctionMessage,
            HumanMessageChunk,
            AIMessageChunk,
            SystemMessageChunk,
            FunctionMessageChunk,
            ChatMessageChunk,
            ChatMessage,
        ]

    messages = [
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

    for msg in messages:
        obj1 = WellKnownTypes.parse_obj(msg.dict())
        assert type(obj1.__root__) == type(msg), f"failed for {type(msg)}"
