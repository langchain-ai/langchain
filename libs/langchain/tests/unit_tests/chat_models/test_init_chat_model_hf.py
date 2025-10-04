import sys
import types
from importlib import util as import_util
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from langchain.chat_models import init_chat_model


@pytest.fixture
def hf_fakes(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Install fakes for Hugging Face and transformers.

    Capture call arguments and simulate module presence to test initialization
    behavior, including current failure modes.
    """
    pipeline_calls: list[tuple[str, dict[str, Any]]] = []
    init_calls: list[dict[str, Any]] = []

    # Fake transformers.pipeline
    def fake_pipeline(task: str, **kwargs: Any) -> SimpleNamespace:
        pipeline_calls.append((task, dict(kwargs)))
        return SimpleNamespace(_kind="dummy_hf_pipeline")

    transformers_mod = types.ModuleType("transformers")
    setattr(transformers_mod, "pipeline", fake_pipeline)  # noqa: B010
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)

    # Fake langchain_huggingface.ChatHuggingFace that REQUIRES `llm`
    class FakeChatHuggingFace:
        def __init__(self, *, llm: object, **kwargs: Any) -> None:
            init_calls.append({"llm": llm, "kwargs": dict(kwargs)})
            self._llm = llm
            self._kwargs = kwargs

    # Build full package path:
    # langchain_huggingface.chat_models.huggingface
    hf_pkg = types.ModuleType("langchain_huggingface")
    hf_pkg.__path__ = []  # mark as package

    hf_chat_models_pkg = types.ModuleType("langchain_huggingface.chat_models")
    hf_chat_models_pkg.__path__ = []  # mark as package

    hf_chat_hf_mod = types.ModuleType(
        "langchain_huggingface.chat_models.huggingface",
    )
    setattr(hf_chat_hf_mod, "ChatHuggingFace", FakeChatHuggingFace)  # noqa: B010

    # Also expose at package root for top-level imports
    setattr(hf_pkg, "ChatHuggingFace", FakeChatHuggingFace)  # noqa: B010

    monkeypatch.setitem(sys.modules, "langchain_huggingface", hf_pkg)
    monkeypatch.setitem(
        sys.modules,
        "langchain_huggingface.chat_models",
        hf_chat_models_pkg,
    )
    monkeypatch.setitem(
        sys.modules,
        "langchain_huggingface.chat_models.huggingface",
        hf_chat_hf_mod,
    )

    # Ensure _check_pkg sees both packages as installed
    orig_find_spec = import_util.find_spec

    def fake_find_spec(name: str) -> Optional[object]:
        if name in {
            "transformers",
            "langchain_huggingface",
            "langchain_huggingface.chat_models",
            "langchain_huggingface.chat_models.huggingface",
        }:
            return object()
        return orig_find_spec(name)

    monkeypatch.setattr("importlib.util.find_spec", fake_find_spec)

    return SimpleNamespace(
        pipeline_calls=pipeline_calls,
        init_calls=init_calls,
    )


def test_hf_current_bug_basic_raises_typeerror(
    hf_fakes: SimpleNamespace,
) -> None:
    """Current behavior raises TypeError when using Hugging Face provider.

    init_chat_model constructs ChatHuggingFace without ``llm`` and never builds
    a pipeline. Verify that explicitly.
    """
    with pytest.raises(TypeError):
        _ = init_chat_model(
            "huggingface:microsoft/Phi-3-mini-4k-instruct",
            task="text-generation",
            temperature=0,
        )
    # Buggy path should not touch transformers.pipeline
    assert not hf_fakes.pipeline_calls, "pipeline should NOT be called"


def test_hf_current_bug_max_tokens_case_raises_typeerror(
    hf_fakes: SimpleNamespace,
) -> None:
    """Same failure when passing ``max_tokens``.

    Should raise and avoid constructing a pipeline.
    """
    with pytest.raises(TypeError):
        _ = init_chat_model(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            model_provider="huggingface",
            task="text-generation",
            max_tokens=42,
        )
    assert not hf_fakes.pipeline_calls, "pipeline should NOT be called"


def test_hf_current_bug_timeout_retries_case_raises_typeerror(
    hf_fakes: SimpleNamespace,
) -> None:
    """Same failure when passing ``timeout``/``max_retries``.

    Should raise and avoid constructing a pipeline.
    """
    with pytest.raises(TypeError):
        _ = init_chat_model(
            model="microsoft/Phi-3-mini-4k-instruct",
            model_provider="huggingface",
            task="text-generation",
            temperature=0.1,
            timeout=7,
            max_retries=3,
        )
    assert not hf_fakes.pipeline_calls, "pipeline should NOT be called"
