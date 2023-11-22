from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
    ToolMessageChunk,
)


def test_lc_namespace() -> None:
    assert BaseMessage.get_lc_namespace() == [
        "langchain",
        "schema",
        "messages",
    ]
    assert AIMessage.get_lc_namespace() == [
        "langchain",
        "schema",
        "messages",
    ]
    assert HumanMessage.get_lc_namespace() == [
        "langchain",
        "schema",
        "messages",
    ]
    assert SystemMessage.get_lc_namespace() == [
        "langchain",
        "schema",
        "messages",
    ]
    assert FunctionMessage.get_lc_namespace() == [
        "langchain",
        "schema",
        "messages",
    ]
    assert ToolMessage.get_lc_namespace() == [
        "langchain",
        "schema",
        "messages",
    ]
    assert ChatMessage.get_lc_namespace() == [
        "langchain",
        "schema",
        "messages",
    ]
    assert BaseMessageChunk.get_lc_namespace() == [
        "langchain",
        "schema",
        "messages",
    ]
    assert AIMessageChunk.get_lc_namespace() == [
        "langchain",
        "schema",
        "messages",
    ]
    assert HumanMessageChunk.get_lc_namespace() == [
        "langchain",
        "schema",
        "messages",
    ]
    assert SystemMessageChunk.get_lc_namespace() == [
        "langchain",
        "schema",
        "messages",
    ]
    assert FunctionMessageChunk.get_lc_namespace() == [
        "langchain",
        "schema",
        "messages",
    ]
    assert ToolMessageChunk.get_lc_namespace() == [
        "langchain",
        "schema",
        "messages",
    ]
    assert ChatMessageChunk.get_lc_namespace() == [
        "langchain",
        "schema",
        "messages",
    ]
