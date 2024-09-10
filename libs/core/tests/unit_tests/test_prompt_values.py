from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
    ToolMessageChunk,
)
from langchain_core.prompt_values import ChatPromptValueConcrete


def test_chat_prompt_value_concrete() -> None:
    messages: list = [
        AIMessage("foo"),
        HumanMessage("foo"),
        SystemMessage("foo"),
        ToolMessage("foo", tool_call_id="foo"),
        AIMessageChunk(content="foo"),
        HumanMessageChunk(content="foo"),
        SystemMessageChunk(content="foo"),
        ToolMessageChunk(content="foo", tool_call_id="foo"),
    ]
    assert ChatPromptValueConcrete(messages=messages).messages == messages
