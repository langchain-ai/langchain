from unittest.mock import MagicMock, patch

import pytest

from langchain_huggingface import HuggingFacePipeline

DEFAULT_MODEL_ID = "gpt2"


def test_initialization_default() -> None:
    """Test default initialization."""
    llm = HuggingFacePipeline()

    assert llm.model_id == DEFAULT_MODEL_ID


@patch("transformers.pipeline")
def test_initialization_with_pipeline(mock_pipeline: MagicMock) -> None:
    """Test initialization with a pipeline object."""
    mock_pipe = MagicMock()
    mock_pipe.model.name_or_path = "mock-model-id"
    mock_pipeline.return_value = mock_pipe

    llm = HuggingFacePipeline(pipeline=mock_pipe)

    assert llm.model_id == "mock-model-id"


@patch("transformers.AutoTokenizer.from_pretrained")
@patch("transformers.AutoModelForCausalLM.from_pretrained")
@patch("transformers.pipeline")
def test_initialization_with_from_model_id(
    mock_pipeline: MagicMock, mock_model: MagicMock, mock_tokenizer: MagicMock
) -> None:
    """Test initialization with the from_model_id method."""
    mock_tokenizer.return_value = MagicMock(pad_token_id=0)
    mock_model.return_value = MagicMock()

    mock_pipe = MagicMock()
    mock_pipe.task = "text-generation"
    mock_pipe.model = mock_model.return_value
    mock_pipeline.return_value = mock_pipe

    llm = HuggingFacePipeline.from_model_id(
        model_id="mock-model-id",
        task="text-generation",
    )

    assert llm.model_id == "mock-model-id"


@patch("transformers.AutoTokenizer.from_pretrained")
@patch("transformers.AutoModelForCausalLM.from_pretrained")
@patch("transformers.pipeline")
def test_from_model_id_uses_torch_accelerator_device_count(
    mock_pipeline: MagicMock, mock_model: MagicMock, mock_tokenizer: MagicMock
) -> None:
    """Test `from_model_id` prefers the device-agnostic `torch.accelerator` API.

    Available in `torch>=2.6`, this is preferred over `torch.cuda` so that
    non-CUDA accelerators, such as XPU, are recognized.
    """
    mock_tokenizer.return_value = MagicMock(pad_token_id=0)
    mock_model.return_value = MagicMock(
        is_loaded_in_4bit=False, is_loaded_in_8bit=False
    )

    mock_pipe = MagicMock()
    mock_pipe.task = "text-generation"
    mock_pipe.model = mock_model.return_value
    mock_pipeline.return_value = mock_pipe

    mock_accelerator = MagicMock()
    mock_accelerator.device_count.return_value = 2

    with (
        patch("torch.accelerator", mock_accelerator),
        patch("torch.cuda.device_count") as mock_cuda_device_count,
    ):
        HuggingFacePipeline.from_model_id(
            model_id="mock-model-id",
            task="text-generation",
            device=0,
        )

    mock_accelerator.device_count.assert_called_once()
    mock_cuda_device_count.assert_not_called()


@patch("transformers.AutoTokenizer.from_pretrained")
@patch("transformers.AutoModelForCausalLM.from_pretrained")
@patch("transformers.pipeline")
def test_from_model_id_falls_back_to_torch_cuda_device_count(
    mock_pipeline: MagicMock,
    mock_model: MagicMock,
    mock_tokenizer: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test `from_model_id` falls back to `torch.cuda.device_count`.

    This happens when the installed `torch` version predates the
    `torch.accelerator` API.
    """
    mock_tokenizer.return_value = MagicMock(pad_token_id=0)
    mock_model.return_value = MagicMock(
        is_loaded_in_4bit=False, is_loaded_in_8bit=False
    )

    mock_pipe = MagicMock()
    mock_pipe.task = "text-generation"
    mock_pipe.model = mock_model.return_value
    mock_pipeline.return_value = mock_pipe

    import torch

    # Simulate a `torch` version predating the `torch.accelerator` API.
    monkeypatch.delattr(torch, "accelerator", raising=False)

    with patch("torch.cuda.device_count", return_value=1) as mock_cuda_device_count:
        HuggingFacePipeline.from_model_id(
            model_id="mock-model-id",
            task="text-generation",
            device=0,
        )

    mock_cuda_device_count.assert_called_once()


@patch("transformers.AutoTokenizer.from_pretrained")
@patch("transformers.AutoModelForCausalLM.from_pretrained")
@patch("transformers.pipeline")
def test_from_model_id_raises_for_out_of_range_device(
    mock_pipeline: MagicMock, mock_model: MagicMock, mock_tokenizer: MagicMock
) -> None:
    """Test `from_model_id` raises when `device` is out of range.

    This is the range reported by the accelerator device count.
    """
    mock_tokenizer.return_value = MagicMock(pad_token_id=0)
    mock_model.return_value = MagicMock(
        is_loaded_in_4bit=False, is_loaded_in_8bit=False
    )

    mock_pipe = MagicMock()
    mock_pipe.task = "text-generation"
    mock_pipe.model = mock_model.return_value
    mock_pipeline.return_value = mock_pipe

    mock_accelerator = MagicMock()
    mock_accelerator.device_count.return_value = 1

    with patch("torch.accelerator", mock_accelerator):
        try:
            HuggingFacePipeline.from_model_id(
                model_id="mock-model-id",
                task="text-generation",
                device=5,
            )
        except ValueError as exc:
            assert "device is required to be within" in str(exc)
        else:
            msg = "Expected ValueError to be raised for out-of-range device"
            raise AssertionError(msg)
