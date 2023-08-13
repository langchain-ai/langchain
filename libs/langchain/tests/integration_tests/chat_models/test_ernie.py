
from langchain.chat_models.ernie import ErnieChat
from libs.langchain.langchain.schema.messages import BaseMessage, HumanMessage

def test_chat_ernie() -> None:
    """Test ErnieChat."""
    chat = ErnieChat()
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)