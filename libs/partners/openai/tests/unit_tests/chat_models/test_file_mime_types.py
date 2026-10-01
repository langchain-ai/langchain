"""Test instance-level file MIME type capabilities."""

from copy import deepcopy
from typing import Any

import pytest
from langchain_core.language_models import ModelProfile
from langchain_core.runnables import RunnableBinding
from pydantic import SecretStr

from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_openai.chat_models.base import _get_default_model_profile
from langchain_openai.data._profiles import _PROFILES


@pytest.mark.parametrize(
    ("model_name", "kwargs", "generic_files"),
    [
        ("gpt-4.1", {}, False),
        ("gpt-4.1", {"use_responses_api": True}, True),
        ("gpt-4.1", {"reasoning": {"summary": "auto"}}, True),
        ("gpt-4.1", {"output_version": "responses/v1"}, True),
        ("gpt-5.2-pro", {}, True),
        ("gpt-5.2-pro", {"use_responses_api": False}, False),
        ("gpt-4.1", {"use_responses_api": False, "reasoning": {}}, False),
    ],
)
def test_file_mime_types_routing(
    model_name: str, kwargs: dict[str, Any], *, generic_files: bool
) -> None:
    model = ChatOpenAI(model=model_name, api_key=SecretStr("test"), **kwargs)
    assert model.profile
    mime_types = model.profile["file_mime_types"]
    assert "application/pdf" in mime_types
    assert {"image/jpeg", "image/png", "image/webp", "image/gif"} <= set(mime_types)
    assert ("text/plain" in mime_types) is generic_files
    assert ("text/csv" in mime_types) is generic_files
    assert not any(
        mime_type.startswith(("audio/", "video/")) for mime_type in mime_types
    )


@pytest.mark.parametrize(
    "model_name",
    [
        "text-embedding-ada-002",
        "text-embedding-3-small",
        "text-embedding-3-large",
        "chatgpt-image-latest",
        "gpt-image-1",
        "gpt-image-1-mini",
        "gpt-image-1.5",
        "gpt-image-2",
        "gpt-realtime-2.1",
    ],
)
def test_non_responses_models_omit_file_mime_types(model_name: str) -> None:
    model = ChatOpenAI(
        model=model_name, api_key=SecretStr("test"), use_responses_api=True
    )
    assert "file_mime_types" not in (model.profile or {})
    assert _PROFILES[model_name]["file_mime_types"] == []


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
    assert second.profile
    assert "text/plain" in second.profile["file_mime_types"]
    assert completions.profile
    assert "application/pdf" in completions.profile["file_mime_types"]
    assert "text/plain" not in completions.profile["file_mime_types"]
    assert _PROFILES["gpt-4.1"] == expected


@pytest.mark.parametrize("use_responses_api", [False, True])
@pytest.mark.parametrize("model_name", ["unknown-model", "gpt-image-2"])
def test_explicit_file_mime_types(model_name: str, *, use_responses_api: bool) -> None:
    profile: ModelProfile = {"file_mime_types": ["application/custom"]}
    model = ChatOpenAI(
        model=model_name,
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
    assert "application/pdf" in model.profile["file_mime_types"]
    assert ("text/plain" in model.profile["file_mime_types"]) is use_responses_api


def test_per_call_routing_does_not_change_profile() -> None:
    model = ChatOpenAI(model="gpt-4.1", api_key=SecretStr("test"))
    bound = model.bind_tools([{"type": "web_search"}])
    assert isinstance(bound, RunnableBinding)
    assert model._use_responses_api(dict(bound.kwargs))
    assert model.profile
    assert "text/plain" not in model.profile["file_mime_types"]


@pytest.mark.parametrize("use_responses_api", [False, True])
def test_unknown_and_text_only_models_omit_file_mime_types(
    *, use_responses_api: bool
) -> None:
    for model_name in ("unknown-model", "gpt-3.5-turbo", "gpt-4"):
        profile = _get_default_model_profile(
            model_name, use_responses_api=use_responses_api
        )
        assert "file_mime_types" not in profile


@pytest.mark.parametrize(
    ("flags", "use_responses_api", "expected"),
    [
        (
            {"pdf_inputs": False, "image_inputs": True},
            False,
            ["image/jpeg", "image/png", "image/webp", "image/gif"],
        ),
        ({"pdf_inputs": True, "image_inputs": False}, False, []),
        ({"pdf_inputs": True, "image_inputs": False}, True, ["text/plain"]),
        ({"audio_inputs": True}, False, ["audio/wav", "audio/mpeg"]),
        ({"audio_inputs": True}, True, ["text/plain"]),
        (
            {"audio_inputs": True, "audio_outputs": True},
            False,
            ["audio/wav", "audio/mpeg"],
        ),
        ({"audio_inputs": True, "audio_outputs": True}, True, []),
        ({"image_inputs": True, "image_outputs": True}, False, []),
        (
            {"image_inputs": True, "tool_calling": False},
            True,
            ["image/jpeg", "image/png", "image/webp", "image/gif"],
        ),
        ({"audio_inputs": False, "video_inputs": True}, False, []),
    ],
)
def test_file_mime_types_follow_modalities_and_transport(
    monkeypatch: pytest.MonkeyPatch,
    flags: dict[str, bool],
    *,
    use_responses_api: bool,
    expected: list[str],
) -> None:
    mime_types = [
        "application/pdf",
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/gif",
        "audio/wav",
        "audio/mpeg",
        "text/plain",
    ]
    profiles = {
        "synthetic": {
            "text_outputs": True,
            "tool_calling": True,
            **flags,
            "file_mime_types": mime_types,
        }
    }
    monkeypatch.setattr("langchain_openai.chat_models.base._MODEL_PROFILES", profiles)
    profile = _get_default_model_profile(
        "synthetic", use_responses_api=use_responses_api
    )
    assert profile.get("file_mime_types", []) == expected
    assert profiles["synthetic"]["file_mime_types"] == mime_types
