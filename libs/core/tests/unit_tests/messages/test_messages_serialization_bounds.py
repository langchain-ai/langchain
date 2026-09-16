\"\"\"
Unit tests for message dictionary conversion and role normalization in langchain_core.messages.
\"\"\"
import pytest
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, messages_to_dict

def test_messages_to_dict_conversion():
    messages = [
        SystemMessage(content="You are helpful."),
        HumanMessage(content="Hello!"),
        AIMessage(content="Hi there!")
    ]
    
    dicts = messages_to_dict(messages)
    assert len(dicts) == 3
    assert dicts[0]["type"] == "system"
    assert dicts[0]["data"]["content"] == "You are helpful."
    assert dicts[1]["type"] == "human"
    assert dicts[1]["data"]["content"] == "Hello!"
    assert dicts[2]["type"] == "ai"
    assert dicts[2]["data"]["content"] == "Hi there!"

def test_empty_messages_to_dict():
    assert messages_to_dict([]) == []