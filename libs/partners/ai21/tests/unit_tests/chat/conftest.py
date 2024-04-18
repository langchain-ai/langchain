import pytest

from langchain_ai21.chat_builder.chat import Chat
from langchain_ai21.chat_builder.chat_factory import create_chat


@pytest.fixture
def chat(model: str) -> Chat:
    return create_chat(model)
