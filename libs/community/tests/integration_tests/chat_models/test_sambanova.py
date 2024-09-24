from langchain_core.messages import AIMessage, HumanMessage

from langchain_community.chat_models.sambanova import ChatSambaNovaCloud


def test_chat_sambanova_cloud() -> None:
    chat = ChatSambaNovaCloud()
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
