"""Test Straico Chat API wrapper."""

import os

import pytest

from langchain_community.chat_models import ChatStraico

os.environ["STRAICO_API_KEY"] = "foo"


@pytest.mark.requires("openai")
def test_straico_model_name_param() -> None:
    llm = ChatStraico(model="foo")
    assert llm.model == "foo"


@pytest.mark.requires("openai")
def test_straico_model_kwargs() -> None:
    llm = ChatStraico(model="test", model_kwargs={"foo": "bar"})
    assert llm.model_kwargs == {"foo": "bar"}


@pytest.mark.requires("openai")
def test_straico_initialization() -> None:
    """Test straico initialization."""
    # Verify that chat straico can be initialized using a secret key provided
    # as a parameter rather than an environment variable.
    for model in [
        ChatStraico(
            model="test",
            timeout=1,
            straico_api_key="test",
            verbose=True,
        ),
        ChatStraico(
            model="test",
            request_timeout=1,
            straico_api_key="test",
            verbose=True,
        ),
    ]:
        assert model.request_timeout == 1
        assert model.straico_api_key == "test"
