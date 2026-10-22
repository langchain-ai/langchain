from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
import torch

# On first import, `transformers.processing_utils` re-executes the `transformers`
# package and replaces it in `sys.modules`. Import it up front so that
# `patch("transformers....")` targets the module `from_model_id` imports from.
import transformers.processing_utils  # noqa: F401

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


@pytest.fixture
def mock_hf_model() -> Iterator[MagicMock]:
    """Patch model and tokenizer loading and yield the mocked model."""
    with (
        patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer,
        patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model,
    ):
        mock_tokenizer.return_value = MagicMock(pad_token_id=0)
        mock_model.return_value = MagicMock(
            is_loaded_in_4bit=False, is_loaded_in_8bit=False
        )
        yield mock_model.return_value


@pytest.fixture
def mock_hf_pipeline(mock_hf_model: MagicMock) -> Iterator[MagicMock]:
    """Patch `transformers.pipeline` and yield the mock."""
    with patch("transformers.pipeline") as mock_pipeline:
        mock_pipeline.return_value = MagicMock(task="text-generation")
        yield mock_pipeline


def _mock_accelerator(accelerator_type: str | None, device_count: int) -> MagicMock:
    mock_accelerator = MagicMock()
    mock_accelerator.device_count.return_value = device_count
    mock_accelerator.current_accelerator.return_value = (
        torch.device(accelerator_type) if accelerator_type else None
    )
    return mock_accelerator


@pytest.mark.parametrize(
    ("accelerator_type", "device"),
    [
        ("xpu", "xpu"),
        ("xpu", "xpu:1"),
        ("xpu", torch.device("xpu", 1)),
        ("cuda", "cuda:1"),
        ("mps", "mps"),
        ("xpu", "cpu"),
        (None, "cpu"),
    ],
)
def test_from_model_id_accepts_device_string_or_torch_device(
    mock_hf_pipeline: MagicMock,
    accelerator_type: str | None,
    device: str | torch.device,
) -> None:
    """Test `from_model_id` passes device strings and `torch.device` through.

    The requested device type is checked against whichever accelerator
    `torch.accelerator` reports, so no backend is special-cased.
    """
    with patch("torch.accelerator", _mock_accelerator(accelerator_type, 2)):
        HuggingFacePipeline.from_model_id(
            model_id="mock-model-id", task="text-generation", device=device
        )

    assert mock_hf_pipeline.call_args.kwargs["device"] == device


@pytest.mark.parametrize(
    ("accelerator_type", "device", "match"),
    [
        ("xpu", "cuda", "the available accelerator is xpu"),
        ("cuda", torch.device("xpu"), "the available accelerator is cuda"),
        (None, "xpu:0", "the available accelerator is None"),
        ("xpu", "xpu:2", r"device index is required to be within \[0, 2\)"),
    ],
)
def test_from_model_id_raises_for_unavailable_device(
    mock_hf_pipeline: MagicMock,
    accelerator_type: str | None,
    device: str | torch.device,
    match: str,
) -> None:
    """Test `from_model_id` rejects devices the accelerator cannot provide."""
    with (
        patch("torch.accelerator", _mock_accelerator(accelerator_type, 2)),
        pytest.raises(ValueError, match=match),
    ):
        HuggingFacePipeline.from_model_id(
            model_id="mock-model-id", task="text-generation", device=device
        )

    mock_hf_pipeline.assert_not_called()


@pytest.mark.parametrize(
    "device_map_kwargs",
    [{"device_map": "xpu:1"}, {"model_kwargs": {"device_map": "xpu:1"}}],
)
def test_from_model_id_keeps_single_device_map_placement(
    mock_hf_model: MagicMock,
    mock_hf_pipeline: MagicMock,
    device_map_kwargs: dict,
) -> None:
    """Test a single-device `device_map` survives pipeline construction.

    Such a map leaves `hf_device_map` unset, so `transformers.pipeline` would
    otherwise move the model to accelerator 0.
    """
    mock_hf_model.hf_device_map = None
    mock_hf_model.device = torch.device("xpu", 1)

    HuggingFacePipeline.from_model_id(
        model_id="mock-model-id", task="text-generation", **device_map_kwargs
    )

    assert mock_hf_pipeline.call_args.kwargs["device"] == torch.device("xpu", 1)


@pytest.mark.parametrize(
    ("device_map", "hf_device_map"),
    [("auto", {"transformer.h.0": 0, "transformer.h.1": 1}), (None, None)],
)
def test_from_model_id_leaves_device_unset_without_single_device_map(
    mock_hf_model: MagicMock,
    mock_hf_pipeline: MagicMock,
    device_map: str | None,
    hf_device_map: dict | None,
) -> None:
    """Test `device` stays unset for `accelerate` maps or when no map is given."""
    mock_hf_model.hf_device_map = hf_device_map
    mock_hf_model.device = torch.device("cpu")

    HuggingFacePipeline.from_model_id(
        model_id="mock-model-id", task="text-generation", device_map=device_map
    )

    assert mock_hf_pipeline.call_args.kwargs["device"] is None
