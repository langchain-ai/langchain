"""Test Contextual AI Client Wrapper."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langchain_community.chat_models import ChatContextual


@pytest.mark.scheduled
def test_contextual_generate() -> None:
    model = ChatContextual(api_key=None)
    response = model.invoke(
        input=[HumanMessage(content="What kind of cats are there?")],
        knowledge=[
            "There are only 2 types of cats in the world, good cats and best cats.",
        ],
    )
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
