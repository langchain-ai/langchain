\"\"\"
Unit tests for StringPromptValue and ChatPromptValue conversion methods in langchain_core.prompt_values.
\"\"\"
import pytest
from langchain_core.prompt_values import StringPromptValue, ChatPromptValue
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

def test_string_prompt_value_methods():
    pv = StringPromptValue(text="Simple text query")
    assert pv.to_string() == "Simple text query"
    messages = pv.to_messages()
    assert len(messages) == 1
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "Simple text query"

def test_chat_prompt_value_methods():
    msgs = [
        SystemMessage(content="You are helpful."),
        HumanMessage(content="Hello!")
    ]
    cpv = ChatPromptValue(messages=msgs)
    assert cpv.to_messages() == msgs
    string_repr = cpv.to_string()
    assert "System: You are helpful." in string_repr
    assert "Human: Hello!" in string_repr