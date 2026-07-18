import pytest
from unittest.mock import MagicMock, patch

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


def _make_mock_transformers_modules(mock_pipeline_fn: MagicMock) -> dict:
    """Return a sys.modules patch dict that stubs out all transformers sub-modules
    needed by from_model_id without requiring a real PyTorch installation."""
    mock_model_instance = MagicMock()
    # Prevent the "device set to None" early-exit path triggered by quantized models.
    mock_model_instance.is_loaded_in_4bit = False
    mock_model_instance.is_loaded_in_8bit = False

    mock_transformers = MagicMock()
    mock_transformers.pipeline = mock_pipeline_fn
    mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = (
        mock_model_instance
    )
    mock_transformers.AutoModelForSeq2SeqLM.from_pretrained.return_value = (
        mock_model_instance
    )
    return {
        "transformers": mock_transformers,
        "transformers.pipelines": MagicMock(),
    }


def test_from_model_id_uses_torch_accelerator_when_available() -> None:
    """from_model_id uses torch.accelerator.device_count() on PyTorch >= 2.6.

    Regression test for https://github.com/langchain-ai/langchain/issues/38855:
    on non-CUDA accelerators (XPU, ROCm) torch.cuda.device_count() returns 0,
    so a valid device index was incorrectly rejected. The fix prefers
    torch.accelerator.device_count() when the API is available.
    """
    mock_pipe = MagicMock()
    mock_pipe.task = "text-generation"
    mock_pipe.model.name_or_path = "mock-model-id"
    mock_pipeline_fn = MagicMock(return_value=mock_pipe)

    mock_torch = MagicMock()
    mock_torch.__spec__ = MagicMock()  # required by importlib.util.find_spec
    mock_torch.cuda.device_count.return_value = 0  # no CUDA devices
    mock_torch.accelerator.device_count.return_value = 1  # 1 XPU/non-CUDA device

    modules = {"torch": mock_torch, **_make_mock_transformers_modules(mock_pipeline_fn)}
    with patch.dict("sys.modules", modules):
        # device=0 is valid for the XPU; must not raise ValueError
        llm = HuggingFacePipeline.from_model_id(
            model_id="mock-model-id",
            task="text-generation",
            device=0,
        )

    assert llm.model_id == "mock-model-id"
    mock_torch.accelerator.device_count.assert_called()
    mock_torch.cuda.device_count.assert_not_called()


def test_from_model_id_falls_back_to_cuda_device_count() -> None:
    """from_model_id falls back to torch.cuda.device_count() on PyTorch < 2.6."""
    mock_pipe = MagicMock()
    mock_pipe.task = "text-generation"
    mock_pipe.model.name_or_path = "mock-model-id"
    mock_pipeline_fn = MagicMock(return_value=mock_pipe)

    # spec=["cuda"] means hasattr(mock_torch, "accelerator") returns False
    mock_torch = MagicMock(spec=["cuda", "__spec__"])
    mock_torch.__spec__ = MagicMock()  # required by importlib.util.find_spec
    mock_torch.cuda.device_count.return_value = 1  # 1 CUDA device

    modules = {"torch": mock_torch, **_make_mock_transformers_modules(mock_pipeline_fn)}
    with patch.dict("sys.modules", modules):
        llm = HuggingFacePipeline.from_model_id(
            model_id="mock-model-id",
            task="text-generation",
            device=0,
        )

    assert llm.model_id == "mock-model-id"
    mock_torch.cuda.device_count.assert_called()


def test_from_model_id_raises_for_out_of_range_device() -> None:
    """from_model_id raises ValueError when device index exceeds available count."""
    mock_pipe = MagicMock()
    mock_pipe.task = "text-generation"
    mock_pipe.model.name_or_path = "mock-model-id"
    mock_pipeline_fn = MagicMock(return_value=mock_pipe)

    mock_torch = MagicMock()
    mock_torch.__spec__ = MagicMock()  # required by importlib.util.find_spec
    mock_torch.accelerator.device_count.return_value = 1  # only device 0 is valid

    modules = {"torch": mock_torch, **_make_mock_transformers_modules(mock_pipeline_fn)}
    with patch.dict("sys.modules", modules):
        with pytest.raises(ValueError, match="device is required to be within"):
            HuggingFacePipeline.from_model_id(
                model_id="mock-model-id",
                task="text-generation",
                device=5,  # out of range
            )
