import os

from langchain_core.messages import AIMessage

from langchain_community.chat_models import MiniMaxChat


def test_chat_minimax_not_group_id() -> None:
    if "MINIMAX_GROUP_ID" in os.environ:
        del os.environ["MINIMAX_GROUP_ID"]
    chat = MiniMaxChat()  # type: ignore[call-arg]
    response = chat.invoke("你好呀")
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_chat_minimax_with_stream() -> None:
    chat = MiniMaxChat()  # type: ignore[call-arg]
    for chunk in chat.stream("你好呀"):
        assert isinstance(chunk, AIMessage)
        assert isinstance(chunk.content, str)
