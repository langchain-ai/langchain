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
@patch("torch.cuda.device_count")
def test_from_model_id_older_pytorch(
    mock_device_count: MagicMock, mock_pipeline: MagicMock, mock_model: MagicMock, mock_tokenizer: MagicMock
) -> None:
    import torch
    
    mock_tokenizer.return_value = MagicMock(pad_token_id=0)
    mock_model.return_value = MagicMock(is_loaded_in_4bit=False, is_loaded_in_8bit=False)
    mock_pipe = MagicMock()
    mock_pipe.task = "text-generation"
    mock_pipe.model = mock_model.return_value
    mock_pipeline.return_value = mock_pipe

    # Simulate older PyTorch by deleting 'accelerator' attribute if it exists
    original_accelerator = None
    if hasattr(torch, "accelerator"):
        original_accelerator = torch.accelerator
        delattr(torch, "accelerator")

    try:
        mock_device_count.return_value = 2

        # Valid device
        HuggingFacePipeline.from_model_id(
            model_id="mock-model-id", task="text-generation", device=0
        )

        # Invalid device
        with pytest.raises(ValueError, match="device is required to be within \\[-1, 2\\)"):
            HuggingFacePipeline.from_model_id(
                model_id="mock-model-id", task="text-generation", device=2
            )
            
        # device=-1
        HuggingFacePipeline.from_model_id(
            model_id="mock-model-id", task="text-generation", device=-1
        )
    finally:
        if original_accelerator is not None:
            torch.accelerator = original_accelerator


@patch("transformers.AutoTokenizer.from_pretrained")
@patch("transformers.AutoModelForCausalLM.from_pretrained")
@patch("transformers.pipeline")
def test_from_model_id_newer_pytorch(
    mock_pipeline: MagicMock, mock_model: MagicMock, mock_tokenizer: MagicMock
) -> None:
    import torch
    
    mock_tokenizer.return_value = MagicMock(pad_token_id=0)
    mock_model.return_value = MagicMock(is_loaded_in_4bit=False, is_loaded_in_8bit=False)
    mock_pipe = MagicMock()
    mock_pipe.task = "text-generation"
    mock_pipe.model = mock_model.return_value
    mock_pipeline.return_value = mock_pipe

    mock_accelerator = MagicMock()
    mock_accelerator.device_count.return_value = 2
    
    with patch.object(torch, "accelerator", mock_accelerator, create=True):
        # Valid device
        HuggingFacePipeline.from_model_id(
            model_id="mock-model-id", task="text-generation", device=1
        )

        # Invalid device
        with pytest.raises(ValueError, match="device is required to be within \\[-1, 2\\)"):
            HuggingFacePipeline.from_model_id(
                model_id="mock-model-id", task="text-generation", device=3
            )


@patch("transformers.AutoTokenizer.from_pretrained")
@patch("transformers.AutoModelForCausalLM.from_pretrained")
@patch("transformers.pipeline")
@patch("langchain_huggingface.llms.huggingface_pipeline.logger.warning")
def test_from_model_id_accelerator_warning(
    mock_warning: MagicMock, mock_pipeline: MagicMock, mock_model: MagicMock, mock_tokenizer: MagicMock
) -> None:
    import torch
    
    mock_tokenizer.return_value = MagicMock(pad_token_id=0)
    mock_model.return_value = MagicMock(is_loaded_in_4bit=False, is_loaded_in_8bit=False)
    mock_pipe = MagicMock()
    mock_pipe.task = "text-generation"
    mock_pipe.model = mock_model.return_value
    mock_pipeline.return_value = mock_pipe

    mock_accelerator = MagicMock()
    mock_accelerator.device_count.return_value = 2
    
    with patch.object(torch, "accelerator", mock_accelerator, create=True):
        HuggingFacePipeline.from_model_id(
            model_id="mock-model-id", task="text-generation", device=-1
        )
        mock_warning.assert_called_once()
        assert "Device has %d accelerators available" in mock_warning.call_args[0][0]
        assert mock_warning.call_args[0][1] == 2
