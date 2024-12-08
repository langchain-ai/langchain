"""Test ChatSnowflakeCortex
Note: This test must be run with the following environment variables set:
    SNOWFLAKE_ACCOUNT="YOUR_SNOWFLAKE_ACCOUNT",
    SNOWFLAKE_USERNAME="YOUR_SNOWFLAKE_USERNAME",
    One of SNOWFLAKE_PASSWORD="YOUR_SNOWFLAKE_PASSWORD" or
    SNOWFLAKE_KEY_FILE="YOUR_SNOWFLAKE_KEY_FILE",
    (Optional) SNOWFLAKE_KEY_FILE_PASSWORD="YOUR_SNOWFLAKE_KEY_FILE_PASSWORD"
"""

import os
from typing import Generator

import pytest
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from langchain_community.chat_models import ChatSnowflakeCortex


@pytest.fixture
def chat() -> Generator[ChatSnowflakeCortex, None, None]:
    connection_params = {
        "account": os.environ["SNOWFLAKE_ACCOUNT"],
        "user": os.environ["SNOWFLAKE_USERNAME"],
    }
    if "SNOWFLAKE_PASSWORD" in os.environ:
        connection_params["password"] = os.environ["SNOWFLAKE_PASSWORD"]
    if "SNOWFLAKE_KEY_FILE":
        connection_params["private_key_file"] = os.environ["SNOWFLAKE_KEY_FILE"]
    if "SNOWFLAKE_KEY_FILE_PASSWORD":
        connection_params["private_key_file_pwd"] = os.environ[
            "SNOWFLAKE_KEY_FILE_PASSWORD"
        ]

    chat_instance = ChatSnowflakeCortex(model="llama3.1-8b")

    yield chat_instance


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
