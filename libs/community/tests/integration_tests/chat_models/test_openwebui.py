"""Test OpenWebUI Chat Model."""

from langchain_core.messages import BaseMessage, HumanMessage

from langchain_community.chat_models.openwebui import OpenWebUIAI


def test_default_call() -> None:
    """Test default model call."""
    llm = OpenWebUIAI(
        temperature=0.1,
        model="gemini-2.0-pro-exp-02-05",
    )
    response = llm.invoke(input=[HumanMessage(content="Hello")])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)
