import sys
import types
from importlib import util as import_util
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from langchain.chat_models import init_chat_model

git add libs/langchain/tests/unit_tests/chat_models/test_init_chat_model_hf.py
@pytest.fixture
def hf_fakes(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """
    Install fake modules for `langchain_huggingface` and `transformers` and
    capture their call arguments for assertions.

    """
    pipeline_calls: list[tuple[str, dict[str, Any]]] = []
    init_calls: list[dict[str, Any]] = []

    # Fake transformers.pipeline
    def fake_pipeline(task: str, **kwargs: Any) -> SimpleNamespace:
        pipeline_calls.append((task, dict(kwargs)))
        # A simple stand-in object for the HF pipeline
        return SimpleNamespace(_kind="dummy_hf_pipeline")

    transformers_mod = types.ModuleType("transformers")
    transformers_mod.pipeline = fake_pipeline
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)

    # Fake langchain_huggingface.ChatHuggingFace that REQUIRES `llm`
    class FakeChatHuggingFace:
        def __init__(self, *, llm: Any, **kwargs: Any) -> None:
            init_calls.append({"llm": llm, "kwargs": dict(kwargs)})
            # minimal instance; tests only assert on ctor args
            self._llm = llm
            self._kwargs = kwargs

    # Build full package path: langchain_huggingface.chat_models.huggingface
    hf_pkg = types.ModuleType("langchain_huggingface")
    hf_pkg.__path__ = []  # mark as package

    hf_chat_models_pkg = types.ModuleType("langchain_huggingface.chat_models")
    hf_chat_models_pkg.__path__ = []  # mark as package

    hf_chat_huggingface_mod = types.ModuleType(
        "langchain_huggingface.chat_models.huggingface"
    )
    hf_chat_huggingface_mod.ChatHuggingFace = FakeChatHuggingFace

    # Optional: expose at package root for compatibility with top-level imports
    hf_pkg.ChatHuggingFace = FakeChatHuggingFace

    monkeypatch.setitem(sys.modules, "langchain_huggingface", hf_pkg)
    monkeypatch.setitem(
        sys.modules,
        "langchain_huggingface.chat_models",
        hf_chat_models_pkg,
    )
    monkeypatch.setitem(
        sys.modules,
        "langchain_huggingface.chat_models.huggingface",
        hf_chat_huggingface_mod,
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

    return SimpleNamespace(pipeline_calls=pipeline_calls, init_calls=init_calls)


def _last_pipeline_kwargs(hf_fakes: SimpleNamespace) -> dict[str, Any]:
    assert hf_fakes.pipeline_calls, "transformers.pipeline was not called"
    _, kwargs = hf_fakes.pipeline_calls[-1]
    return kwargs


def _last_chat_kwargs(hf_fakes: SimpleNamespace) -> dict[str, Any]:
    assert hf_fakes.init_calls, "ChatHuggingFace was not constructed"
    return hf_fakes.init_calls[-1]["kwargs"]


@pytest.mark.xfail(
    reason=(
        "Pending fix for huggingface init (#28226 / #33167) — currently passes "
        "model_id to ChatHuggingFace"
    ),
    raises=TypeError,
)
def test_hf_basic_wraps_pipeline(hf_fakes: SimpleNamespace) -> None:
    # provider specified inline
    llm = init_chat_model(
        "huggingface:microsoft/Phi-3-mini-4k-instruct",
        task="text-generation",
        temperature=0,
    )
    # Wrapped object should be constructed (we don't require a specific type here)
    assert llm is not None

    # Make failure modes explicit
    assert hf_fakes.pipeline_calls, "Expected transformers.pipeline to be called"
    assert hf_fakes.init_calls, "Expected ChatHuggingFace to be constructed"

    # pipeline called with correct model (don't assert task value)
    kwargs = _last_pipeline_kwargs(hf_fakes)
    assert kwargs["model"] == "microsoft/Phi-3-mini-4k-instruct"

    # ChatHuggingFace must be constructed with llm
    assert "llm" in hf_fakes.init_calls[-1]
    assert hf_fakes.init_calls[-1]["llm"]._kind == "dummy_hf_pipeline"


@pytest.mark.xfail(
    reason="Pending fix for huggingface init (#28226 / #33167)",
    raises=TypeError,
)
def test_hf_max_tokens_translated_to_max_new_tokens(
    hf_fakes: SimpleNamespace,
) -> None:
    init_chat_model(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        model_provider="huggingface",
        task="text-generation",
        max_tokens=42,
    )
    assert hf_fakes.pipeline_calls, "Expected transformers.pipeline to be called"
    assert hf_fakes.init_calls, "Expected ChatHuggingFace to be constructed"
    kwargs = _last_pipeline_kwargs(hf_fakes)
    assert kwargs.get("max_new_tokens") == 42
    # Ensure we don't leak the old name into pipeline kwargs
    assert "max_tokens" not in kwargs


@pytest.mark.xfail(
    reason="Pending fix for huggingface init (#28226 / #33167)",
    raises=TypeError,
)
def test_hf_timeout_and_max_retries_pass_through_to_chat_wrapper(
    hf_fakes: SimpleNamespace,
) -> None:
    init_chat_model(
        model="microsoft/Phi-3-mini-4k-instruct",
        model_provider="huggingface",
        task="text-generation",
        temperature=0.1,
        timeout=7,
        max_retries=3,
    )
    assert hf_fakes.pipeline_calls, "Expected transformers.pipeline to be called"
    assert hf_fakes.init_calls, "Expected ChatHuggingFace to be constructed"
    chat_kwargs = _last_chat_kwargs(hf_fakes)
    # Assert these control args are passed to the wrapper (not the pipeline)
    assert chat_kwargs.get("timeout") == 7
    assert chat_kwargs.get("max_retries") == 3
    # And that they are NOT passed to transformers.pipeline
    pipeline_kwargs = _last_pipeline_kwargs(hf_fakes)
    assert "timeout" not in pipeline_kwargs
    assert "max_retries" not in pipeline_kwargs
