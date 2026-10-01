"""Test instance-level file MIME type capabilities."""

from copy import deepcopy
from typing import Any

import pytest
from langchain_core.language_models import ModelProfile
from langchain_core.runnables import RunnableBinding
from pydantic import SecretStr

from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_openai.data._profiles import _PROFILES


@pytest.mark.parametrize(
    ("model_name", "kwargs", "supported"),
    [
        ("gpt-4.1", {}, False),
        ("gpt-4.1", {"use_responses_api": True}, True),
        ("gpt-4.1", {"reasoning": {"summary": "auto"}}, True),
        ("gpt-4.1", {"output_version": "responses/v1"}, True),
        ("gpt-5.2-pro", {}, True),
        ("gpt-5.2-pro", {"use_responses_api": False}, False),
        ("gpt-4.1", {"use_responses_api": False, "reasoning": {}}, False),
        ("gpt-3.5-turbo", {"use_responses_api": True}, False),
        ("unknown-model", {"use_responses_api": True}, False),
    ],
)
def test_file_mime_types_routing(
    model_name: str, kwargs: dict[str, Any], *, supported: bool
) -> None:
    model = ChatOpenAI(model=model_name, api_key=SecretStr("test"), **kwargs)
    if supported:
        assert model.profile
        assert (
            model.profile["file_mime_types"] == _PROFILES[model_name]["file_mime_types"]
        )
        assert "application/pdf" not in model.profile["file_mime_types"]
        assert "text/plain" in model.profile["file_mime_types"]
    else:
        assert "file_mime_types" not in (model.profile or {})


def test_file_mime_types_isolation() -> None:
    expected = deepcopy(_PROFILES["gpt-4.1"])
    first = ChatOpenAI(
        model="gpt-4.1", api_key=SecretStr("test"), use_responses_api=True
    )
    assert first.profile
    first.profile["file_mime_types"].clear()
    second = ChatOpenAI(
        model="gpt-4.1", api_key=SecretStr("test"), use_responses_api=True
    )
    completions = ChatOpenAI(model="gpt-4.1", api_key=SecretStr("test"))
    assert second.profile == expected
    assert "file_mime_types" not in (completions.profile or {})
    assert _PROFILES["gpt-4.1"] == expected


@pytest.mark.parametrize("use_responses_api", [False, True])
def test_explicit_file_mime_types(*, use_responses_api: bool) -> None:
    profile: ModelProfile = {"file_mime_types": ["application/custom"]}
    model = ChatOpenAI(
        model="unknown-model",
        api_key=SecretStr("test"),
        use_responses_api=use_responses_api,
        profile=profile,
    )
    assert model.profile == profile


@pytest.mark.parametrize("use_responses_api", [False, True])
@pytest.mark.parametrize("model_name", [None, "gpt-4.1", "unknown-model"])
def test_azure_file_mime_types(
    model_name: str | None, *, use_responses_api: bool
) -> None:
    model = AzureChatOpenAI(
        model=model_name,
        azure_deployment="gpt-4.1",
        azure_endpoint="https://example.openai.azure.com",
        api_version="2025-04-01-preview",
        api_key=SecretStr("test"),
        use_responses_api=use_responses_api,
    )
    assert model.profile
    assert ("file_mime_types" in model.profile) is use_responses_api


def test_per_call_routing_does_not_change_profile() -> None:
    model = ChatOpenAI(model="gpt-4.1", api_key=SecretStr("test"))
    bound = model.bind_tools([{"type": "web_search"}])
    assert isinstance(bound, RunnableBinding)
    assert model._use_responses_api(dict(bound.kwargs))
    assert "file_mime_types" not in (model.profile or {})
