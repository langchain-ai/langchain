from langchain_core.messages import AIMessage, HumanMessage

from langchain_community.chat_models.octoai import ChatOctoAI


def test_chat_octoai() -> None:
    chat = ChatOctoAI()
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
