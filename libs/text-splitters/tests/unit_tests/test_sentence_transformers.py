import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from langchain_text_splitters import SentenceTransformersTokenTextSplitter


@pytest.fixture
def mock_sentence_transformer(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    sentence_transformer_cls = MagicMock()
    sentence_transformer_cls.return_value.max_seq_length = 128
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=sentence_transformer_cls),
    )
    return sentence_transformer_cls


def test_loads_model_on_cpu_by_default(mock_sentence_transformer: MagicMock) -> None:
    """Test the model is loaded on CPU, since only the tokenizer is used."""
    SentenceTransformersTokenTextSplitter(model_name="mock-model")

    mock_sentence_transformer.assert_called_once_with("mock-model", device="cpu")


def test_respects_explicit_device(mock_sentence_transformer: MagicMock) -> None:
    """Test an explicit `device` in `model_kwargs` overrides the CPU default."""
    SentenceTransformersTokenTextSplitter(
        model_name="mock-model", model_kwargs={"device": "cuda:1"}
    )

    mock_sentence_transformer.assert_called_once_with("mock-model", device="cuda:1")


def test_forwards_other_model_kwargs(mock_sentence_transformer: MagicMock) -> None:
    """Test other `model_kwargs` are still forwarded alongside the CPU default."""
    SentenceTransformersTokenTextSplitter(
        model_name="mock-model", model_kwargs={"local_files_only": True}
    )

    mock_sentence_transformer.assert_called_once_with(
        "mock-model", device="cpu", local_files_only=True
    )
