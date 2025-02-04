from langchain_core.messages import AIMessage, HumanMessage

from langchain_community.chat_models.aibrary import ChatAiBrary


def test_chat_aibrary() -> None:
    chat = ChatAiBrary()
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)

def test_get_all_models_aibrary() -> None:
    chat = ChatAiBrary()
    response = chat.get_available_models()
    assert isinstance(response, set)
