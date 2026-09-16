\"\"\"
Unit tests for ChatPromptTemplate message tuple instantiation and partial variable substitution in langchain_core.prompts.
\"\"\"
import pytest
from langchain_core.prompts import ChatPromptTemplate

def test_chat_prompt_template_from_messages_tuples():
    template = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant for {domain}."),
        ("human", "Help me with {task}."),
    ])
    
    formatted = template.format_messages(domain="coding", task="refactoring")
    assert len(formatted) == 2
    assert formatted[0].content == "You are an assistant for coding."
    assert formatted[1].content == "Help me with refactoring."

def test_chat_prompt_template_partial_variables():
    template = ChatPromptTemplate.from_messages([
        ("system", "System name: {bot_name}"),
        ("human", "{user_input}"),
    ])
    
    partial_template = template.partial(bot_name="LangBot")
    formatted = partial_template.format_messages(user_input="Hello!")
    assert formatted[0].content == "System name: LangBot"
    assert formatted[1].content == "Hello!"