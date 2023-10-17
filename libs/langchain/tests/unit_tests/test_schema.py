"""Test formatting functionality."""

import unittest
from typing import Union

from langchain.prompts.base import StringPromptValue
from langchain.prompts.chat import ChatPromptValueConcrete
from langchain.pydantic_v1 import BaseModel
from langchain.schema import AgentAction, AgentFinish, Document
from langchain.schema.agent import AgentActionMessageLog
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
    ]

    for lc_object in lc_objects:
        d = lc_object.dict()
        assert "type" in d, f"Missing key `type` for {type(lc_object)}"
        obj1 = WellKnownLCObject.parse_obj(d)
        assert type(obj1.__root__) == type(lc_object), f"failed for {type(lc_object)}"
