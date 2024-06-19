import pytest

from langchain_ai21.chat.chat_adapter import ChatAdapter
from langchain_ai21.chat.chat_factory import create_chat_adapter


@pytest.fixture
def chat_adapter(model: str) -> ChatAdapter:
    return create_chat_adapter(model)
