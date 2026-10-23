from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from langchain_huggingface import HuggingFaceEmbeddings


@pytest.fixture
def mock_client() -> Iterator[MagicMock]:
    """Patch `SentenceTransformer` and yield the mocked model."""
    with patch("sentence_transformers.SentenceTransformer") as mock_cls:
        mock_cls.return_value.encode.return_value = np.array([[0.1, 0.2]])
        yield mock_cls.return_value


def _mock_accelerator(accelerator_type: str | None, device_count: int) -> MagicMock:
    mock_accelerator = MagicMock()
    mock_accelerator.device_count.return_value = device_count
    mock_accelerator.current_accelerator.return_value = (
        torch.device(accelerator_type) if accelerator_type else None
    )
    return mock_accelerator


@pytest.mark.parametrize(
    ("accelerator_type", "expected_devices"),
    [
        ("xpu", ["xpu:0", "xpu:1"]),
        ("cuda", ["cuda:0", "cuda:1"]),
        (None, None),
    ],
)
def test_multi_process_uses_accelerator_devices(
    mock_client: MagicMock,
    accelerator_type: str | None,
    expected_devices: list[str] | None,
) -> None:
    """Test the multi-process pool gets one worker per `torch.accelerator` device.

    Without an accelerator, `sentence-transformers` picks its own default.
    """
    embeddings = HuggingFaceEmbeddings(
        multi_process=True, encode_kwargs={"normalize_embeddings": True}
    )

    with patch("torch.accelerator", _mock_accelerator(accelerator_type, 2)):
        result = embeddings.embed_documents(["hello"])

    assert result == [[0.1, 0.2]]
    mock_client.start_multi_process_pool.assert_called_once_with(expected_devices)
    pool = mock_client.start_multi_process_pool.return_value
    mock_client.encode.assert_called_once_with(
        ["hello"], pool=pool, show_progress_bar=False, normalize_embeddings=True
    )
    mock_client.stop_multi_process_pool.assert_called_once_with(pool)


def test_multi_process_stops_pool_on_error(mock_client: MagicMock) -> None:
    """Test the multi-process pool is stopped even if encoding fails."""
    mock_client.encode.side_effect = RuntimeError("boom")
    embeddings = HuggingFaceEmbeddings(multi_process=True)

    with pytest.raises(RuntimeError, match="boom"):
        embeddings.embed_query("hello")

    mock_client.stop_multi_process_pool.assert_called_once_with(
        mock_client.start_multi_process_pool.return_value
    )


@pytest.mark.parametrize(
    ("xpu_available", "expected_devices"),
    [(True, ["xpu:0", "xpu:1"]), (False, None)],
)
def test_multi_process_falls_back_to_xpu_without_torch_accelerator(
    mock_client: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    xpu_available: bool,  # noqa: FBT001
    expected_devices: list[str] | None,
) -> None:
    """Test XPU devices are still used when `torch.accelerator` is missing.

    `sentence-transformers` already detects CUDA and NPU devices on its own.
    """
    # Simulate a `torch` version predating the `torch.accelerator` API.
    monkeypatch.delattr(torch, "accelerator", raising=False)
    embeddings = HuggingFaceEmbeddings(multi_process=True)

    with (
        patch("torch.xpu.is_available", return_value=xpu_available),
        patch("torch.xpu.device_count", return_value=2),
    ):
        embeddings.embed_query("hello")

    mock_client.start_multi_process_pool.assert_called_once_with(expected_devices)
