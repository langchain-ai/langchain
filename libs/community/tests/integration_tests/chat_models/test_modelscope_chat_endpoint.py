"""Test Modelscope Chat Model API."""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from langchain_community.chat_models.modelscope_endpoint import ModelscopeChatEndpoint


def test_modelscope_chat_call() -> None:
    chat = ModelscopeChatEndpoint(model="Qwen/Qwen2.5-Coder-32B-Instruct")
    response = chat.invoke([HumanMessage(content="Say foo:")])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_modelscope_chat_multiple_history() -> None:
    """Tests multiple history works."""
    chat = ModelscopeChatEndpoint(model="Qwen/Qwen2.5-Coder-32B-Instruct")

    response = chat.invoke(
        [
            HumanMessage(content="Hello."),
            AIMessage(content="Hello!"),
            HumanMessage(content="How are you doing?"),
        ]
    )
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_modelscope_chat_stream() -> None:
    """Test that stream works."""
    chat = ModelscopeChatEndpoint(
        model="Qwen/Qwen2.5-Coder-32B-Instruct",
        streaming=True,
    )
    response = chat.stream(
        [
            HumanMessage(content="Hello."),
            AIMessage(content="Hello!"),
            HumanMessage(content="Who are you?"),
        ]
    )
    for chunk in response:
        assert isinstance(chunk.content, str)
