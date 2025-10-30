"""End to end test for fetching model profiles from a chat model."""

import pytest

from langchain.chat_models import init_chat_model


def test_chat_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that chat model gets profile data correctly."""
    fake_data = {"openai": {"models": {"gpt-5": {"limit": {"context": 1024}}}}}

    monkeypatch.setattr(
        "langchain.model_profiles._models_dev_sdk._ModelsDevClient._data",
        fake_data,
    )

    model = init_chat_model("openai:gpt-5", api_key="foo")
    assert model.profile
    assert model.profile["max_input_tokens"] == 1024


def test_chat_model_no_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that chat model handles missing profile data."""
    fake_data = {"openai": {"models": {"gpt-5": {"limit": {"context": 1024}}}}}

    monkeypatch.setattr(
        "langchain.model_profiles._models_dev_sdk._ModelsDevClient._data",
        fake_data,
    )

    model = init_chat_model("openai:gpt-fake", api_key="foo")
    assert model.profile == {}
