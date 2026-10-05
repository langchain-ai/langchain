"""Regression tests for Ollama client kwargs ownership."""

from unittest.mock import MagicMock, patch

import pytest

from langchain_ollama import ChatOllama, OllamaEmbeddings, OllamaLLM
from langchain_ollama._utils import merge_auth_headers


@pytest.mark.parametrize(
    "model_cls,patch_target",
    [
        (ChatOllama, "langchain_ollama.chat_models"),
        (OllamaLLM, "langchain_ollama.llms"),
        (OllamaEmbeddings, "langchain_ollama.embeddings"),
    ],
)
def test_url_auth_does_not_mutate_client_kwargs(model_cls, patch_target):
    """URL credentials must not mutate caller-owned client kwargs."""
    client_kwargs = {"headers": {"X-Tenant": "acme"}}
    original_headers = dict(client_kwargs["headers"])

    with patch(f"{patch_target}.Client") as client_cls, patch(
        f"{patch_target}.AsyncClient"
    ) as async_client_cls:
        client_cls.return_value = MagicMock()
        async_client_cls.return_value = MagicMock()
        kwargs = {
            "model": "test-model",
            "base_url": "https://alice:secret@example.com:11434",
            "client_kwargs": client_kwargs,
        }
        model_cls(**kwargs)

    assert client_kwargs == {"headers": original_headers}
    assert "Authorization" not in client_kwargs["headers"]

    passed_kwargs = client_cls.call_args.kwargs
    assert passed_kwargs["headers"]["X-Tenant"] == "acme"
    assert passed_kwargs["headers"]["Authorization"].startswith("Basic ")


def test_merge_auth_headers_does_not_mutate_nested_headers():
    """The auth helper must not mutate a caller-owned nested headers mapping."""
    headers = {"X-Tenant": "acme"}
    client_kwargs = {"headers": headers}

    merge_auth_headers(client_kwargs, {"Authorization": "Basic dGVzdDpzZWNyZXQ="})

    assert headers == {"X-Tenant": "acme"}
    assert client_kwargs["headers"] == {
        "X-Tenant": "acme",
        "Authorization": "Basic dGVzdDpzZWNyZXQ=",
    }
