"""Test ChatSnowflakeCortex
Note: This test must be run with the following environment variables set:
    SNOWFLAKE_ACCOUNT="YOUR_SNOWFLAKE_ACCOUNT",
    SNOWFLAKE_USERNAME="YOUR_SNOWFLAKE_USERNAME",
    SNOWFLAKE_PASSWORD="YOUR_SNOWFLAKE_PASSWORD",
    SNOWFLAKE_DATABASE="YOUR_SNOWFLAKE_DATABASE",
    SNOWFLAKE_SCHEMA="YOUR_SNOWFLAKE_SCHEMA",
    SNOWFLAKE_WAREHOUSE="YOUR_SNOWFLAKE_WAREHOUSE"
    SNOWFLAKE_ROLE="YOUR_SNOWFLAKE_ROLE",
"""

import pytest
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from langchain_community.chat_models import ChatSnowflakeCortex


@pytest.fixture
def chat() -> ChatSnowflakeCortex:
    return ChatSnowflakeCortex()


def test_chat_snowflake_cortex(chat: ChatSnowflakeCortex) -> None:
    """Test ChatSnowflakeCortex."""
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_chat_snowflake_cortex_system_message(chat: ChatSnowflakeCortex) -> None:
    """Test ChatSnowflakeCortex for system message"""
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    response = chat([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_chat_snowflake_cortex_model() -> None:
    """Test ChatSnowflakeCortex handles model_name."""
    chat = ChatSnowflakeCortex(
        model="foo",
    )
    assert chat.model == "foo"


def test_chat_snowflake_cortex_generate(chat: ChatSnowflakeCortex) -> None:
    """Test ChatSnowflakeCortex with generate."""
    message = HumanMessage(content="Hello")
    response = chat.generate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content
