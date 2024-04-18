import pytest

from langchain_ai21.chat_builder.chat_builder import ChatBuilder
from langchain_ai21.chat_builder.chat_builder_factory import create_chat_builder


@pytest.fixture
def chat_builder(model: str) -> ChatBuilder:
    return create_chat_builder(model)
