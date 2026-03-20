"""Unit tests for HuggingFaceEmbeddings (no network calls)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings


def _make_model(**kwargs: Any) -> HuggingFaceEmbeddings:
    """Build HuggingFaceEmbeddings with sentence_transformers mocked out."""
    with patch("sentence_transformers.SentenceTransformer"):
        return HuggingFaceEmbeddings(**kwargs)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_default_batch_size() -> None:
    model = _make_model()
    assert model.batch_size == 32


def test_custom_batch_size() -> None:
    model = _make_model(batch_size=128)
    assert model.batch_size == 128


def test_default_construction_has_no_device_in_model_kwargs() -> None:
    """Default construction must not force device='cpu'.

    sentence-transformers should auto-select the best backend
    (MPS on Apple Silicon, CUDA on NVIDIA).
    """
    model = _make_model()
    assert "device" not in model.model_kwargs


# ---------------------------------------------------------------------------
# convert_to_tensor default
# ---------------------------------------------------------------------------


def test_convert_to_tensor_passed_by_default() -> None:
    """encode() should receive convert_to_tensor=True by default.

    This keeps batch outputs on device and avoids per-batch device→CPU
    synchronisation.
    """
    with patch("sentence_transformers.SentenceTransformer") as mock_cls:
        mock_client = MagicMock()
        mock_client.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_cls.return_value = mock_client
        model = HuggingFaceEmbeddings()

    model.embed_documents(["hello"])

    kwargs = mock_client.encode.call_args.kwargs
    assert kwargs.get("convert_to_tensor") is True


def test_encode_kwargs_can_override_convert_to_tensor() -> None:
    """Users must be able to opt out of the convert_to_tensor default.

    Passing encode_kwargs={'convert_to_tensor': False} must take effect.
    """
    with patch("sentence_transformers.SentenceTransformer") as mock_cls:
        mock_client = MagicMock()
        mock_client.encode.return_value = np.array([[0.1, 0.2]])
        mock_cls.return_value = mock_client
        model = HuggingFaceEmbeddings(encode_kwargs={"convert_to_tensor": False})

    model.embed_documents(["hello"])

    kwargs = mock_client.encode.call_args.kwargs
    assert kwargs.get("convert_to_tensor") is False


# ---------------------------------------------------------------------------
# batch_size forwarding
# ---------------------------------------------------------------------------


def test_batch_size_forwarded_to_encode() -> None:
    with patch("sentence_transformers.SentenceTransformer") as mock_cls:
        mock_client = MagicMock()
        mock_client.encode.return_value = np.array([[0.1, 0.2]])
        mock_cls.return_value = mock_client
        model = HuggingFaceEmbeddings(batch_size=64)

    model.embed_documents(["hello"])

    kwargs = mock_client.encode.call_args.kwargs
    assert kwargs["batch_size"] == 64


def test_encode_kwargs_batch_size_overrides_field() -> None:
    """encode_kwargs batch_size must win over the field-level default."""
    with patch("sentence_transformers.SentenceTransformer") as mock_cls:
        mock_client = MagicMock()
        mock_client.encode.return_value = np.array([[0.1, 0.2]])
        mock_cls.return_value = mock_client
        model = HuggingFaceEmbeddings(
            batch_size=32,
            encode_kwargs={"batch_size": 16},
        )

    model.embed_documents(["hello"])

    kwargs = mock_client.encode.call_args.kwargs
    assert kwargs["batch_size"] == 16


# ---------------------------------------------------------------------------
# Tensor vs numpy conversion path
# ---------------------------------------------------------------------------


def test_torch_tensor_result_uses_cpu_numpy_tolist_path() -> None:
    """When encode() returns a torch tensor, _embed uses cpu().numpy().tolist().

    With convert_to_tensor=True (the default), _embed must call
    .cpu().numpy().tolist() — not tensor.tolist() — to stay on numpy's
    C-optimised conversion path.
    """
    mock_tensor = MagicMock()
    mock_numpy = np.array([[0.1, 0.2], [0.3, 0.4]])
    mock_tensor.cpu.return_value.numpy.return_value = mock_numpy

    with patch("sentence_transformers.SentenceTransformer") as mock_cls:
        mock_client = MagicMock()
        mock_client.encode.return_value = mock_tensor
        mock_cls.return_value = mock_client
        model = HuggingFaceEmbeddings()

    result = model.embed_documents(["a", "b"])

    mock_tensor.cpu.assert_called_once()
    mock_tensor.cpu.return_value.numpy.assert_called_once()
    assert result == mock_numpy.tolist()


def test_numpy_array_result_uses_tolist_directly() -> None:
    """When encode() returns a numpy array, _embed calls .tolist() directly.

    This happens when the user sets convert_to_tensor=False; the numpy
    path must not call .cpu().
    """
    array = np.array([[0.5, 0.6]])

    with patch("sentence_transformers.SentenceTransformer") as mock_cls:
        mock_client = MagicMock()
        mock_client.encode.return_value = array
        mock_cls.return_value = mock_client
        model = HuggingFaceEmbeddings(encode_kwargs={"convert_to_tensor": False})

    result = model.embed_documents(["hello"])

    assert result == [[0.5, 0.6]]


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------


def test_newlines_replaced_before_encoding() -> None:
    with patch("sentence_transformers.SentenceTransformer") as mock_cls:
        mock_client = MagicMock()
        mock_client.encode.return_value = np.array([[0.1, 0.2]])
        mock_cls.return_value = mock_client
        model = HuggingFaceEmbeddings()

    model.embed_documents(["hello\nworld"])

    texts_passed = mock_client.encode.call_args.args[0]
    assert texts_passed == ["hello world"]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_raises_type_error_when_encode_returns_list() -> None:
    with patch("sentence_transformers.SentenceTransformer") as mock_cls:
        mock_client = MagicMock()
        mock_client.encode.return_value = [[0.1, 0.2]]  # list, not array/tensor
        mock_cls.return_value = mock_client
        model = HuggingFaceEmbeddings()

    with pytest.raises(TypeError, match="got a list instead"):
        model.embed_documents(["hello"])


# ---------------------------------------------------------------------------
# query_encode_kwargs fallback
# ---------------------------------------------------------------------------


def test_embed_query_uses_query_encode_kwargs_when_non_empty() -> None:
    with patch("sentence_transformers.SentenceTransformer") as mock_cls:
        mock_client = MagicMock()
        mock_client.encode.return_value = np.array([[0.1, 0.2]])
        mock_cls.return_value = mock_client
        model = HuggingFaceEmbeddings(
            encode_kwargs={"normalize_embeddings": False},
            query_encode_kwargs={"normalize_embeddings": True},
        )

    model.embed_query("test")

    kwargs = mock_client.encode.call_args.kwargs
    assert kwargs["normalize_embeddings"] is True


def test_embed_query_falls_back_to_encode_kwargs_when_query_kwargs_empty() -> None:
    with patch("sentence_transformers.SentenceTransformer") as mock_cls:
        mock_client = MagicMock()
        mock_client.encode.return_value = np.array([[0.1, 0.2]])
        mock_cls.return_value = mock_client
        model = HuggingFaceEmbeddings(
            encode_kwargs={"normalize_embeddings": False},
        )

    model.embed_query("test")

    kwargs = mock_client.encode.call_args.kwargs
    assert kwargs["normalize_embeddings"] is False
