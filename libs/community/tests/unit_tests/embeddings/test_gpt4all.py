import sys
from typing import Any, Optional
from unittest.mock import MagicMock, patch

from langchain_community.embeddings import GPT4AllEmbeddings

_GPT4ALL_MODEL_NAME = "all-MiniLM-L6-v2.gguf2.f16.gguf"
_GPT4ALL_NTHREADS = 4
_GPT4ALL_DEVICE = "gpu"
_GPT4ALL_KWARGS = {"allow_download": False}


class MockEmbed4All(MagicMock):
    """Mock Embed4All class."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        *,
        n_threads: Optional[int] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ):  # type: ignore[no-untyped-def]
        assert model_name == _GPT4ALL_MODEL_NAME


class MockGpt4AllPackage(MagicMock):
    """Mock gpt4all package."""

    Embed4All = MockEmbed4All


def test_create_gpt4all_embeddings_no_kwargs() -> None:
    """Test fix for #25119"""
    with patch.dict(sys.modules, {"gpt4all": MockGpt4AllPackage()}):
        embedding = GPT4AllEmbeddings(  # type: ignore[call-arg]
            model_name=_GPT4ALL_MODEL_NAME,
            n_threads=_GPT4ALL_NTHREADS,
            device=_GPT4ALL_DEVICE,
        )

        assert embedding.model_name == _GPT4ALL_MODEL_NAME
        assert embedding.n_threads == _GPT4ALL_NTHREADS
        assert embedding.device == _GPT4ALL_DEVICE
        assert embedding.gpt4all_kwargs == {}
        assert isinstance(embedding.client, MockEmbed4All)


def test_create_gpt4all_embeddings_with_kwargs() -> None:
    with patch.dict(sys.modules, {"gpt4all": MockGpt4AllPackage()}):
        embedding = GPT4AllEmbeddings(  # type: ignore[call-arg]
            model_name=_GPT4ALL_MODEL_NAME,
            n_threads=_GPT4ALL_NTHREADS,
            device=_GPT4ALL_DEVICE,
            gpt4all_kwargs=_GPT4ALL_KWARGS,
        )

        assert embedding.model_name == _GPT4ALL_MODEL_NAME
        assert embedding.n_threads == _GPT4ALL_NTHREADS
        assert embedding.device == _GPT4ALL_DEVICE
        assert embedding.gpt4all_kwargs == _GPT4ALL_KWARGS
        assert isinstance(embedding.client, MockEmbed4All)
