"""Test JinaChat Chat API wrapper."""

import os

import pytest

from langchain_community.chat_models import JinaChat

os.environ["JINACHAT_API_KEY"] = "foo"


@pytest.mark.requires("openai")
def test_jinachat_model_name_param() -> None:
    llm = JinaChat(model="foo")  # type: ignore[call-arg]
    assert llm.model_kwargs["model"] == "foo"


@pytest.mark.requires("openai")
def test_jinachat_model_kwargs() -> None:
    llm = JinaChat(model="test", model_kwargs={"foo": "bar"})  # type: ignore[call-arg]
    expected_kwargs = {"foo": "bar", "model": "test"}
    assert llm.model_kwargs == expected_kwargs


@pytest.mark.requires("openai")
def test_jinachat_initialization() -> None:
    """Test jinachat initialization."""
    for model in [
        JinaChat(  # type: ignore[call-arg]
            model="test", timeout=1, api_key="test", temperature=0.7, verbose=True
        ),
        JinaChat(  # type: ignore[call-arg]
            model="test",
            request_timeout=1,
            jinachat_api_key="test",
            temperature=0.7,
            verbose=True,
        ),
    ]:
        assert model.request_timeout == 1
        assert model.jinachat_api_key.get_secret_value() == "test"
