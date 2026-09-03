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


def test_generate_translation_task() -> None:
    """Translation pipelines report a task like ``translation_en_to_fr``.

    The output payload uses the ``translation_text`` key, so ``_generate`` must
    route these tasks to that key rather than rejecting them as invalid.
    """
    mock_pipe = MagicMock()
    mock_pipe.task = "translation_en_to_fr"
    mock_pipe.model.name_or_path = "mock-model-id"
    mock_pipe.return_value = [{"translation_text": "Bonjour le monde"}]

    llm = HuggingFacePipeline(pipeline=mock_pipe)
    result = llm._generate(["Hello world"])

    assert result.generations[0][0].text == "Bonjour le monde"


def test_generate_bare_translation_task() -> None:
    """A pipeline whose task is exactly ``translation`` must also be handled."""
    mock_pipe = MagicMock()
    mock_pipe.task = "translation"
    mock_pipe.model.name_or_path = "mock-model-id"
    mock_pipe.return_value = [{"translation_text": "Bonjour"}]

    llm = HuggingFacePipeline(pipeline=mock_pipe)
    result = llm._generate(["Hello"])

    assert result.generations[0][0].text == "Bonjour"


def test_generate_none_or_invalid_task_raises_value_error() -> None:
    """Pipelines with None or invalid task must raise ValueError, not AttributeError."""
    import pytest

    # Test None task
    mock_pipe_none = MagicMock()
    mock_pipe_none.task = None
    mock_pipe_none.model.name_or_path = "mock-model-id"
    mock_pipe_none.return_value = [{"text": "Something"}]

    llm_none = HuggingFacePipeline(pipeline=mock_pipe_none)
    with pytest.raises(ValueError, match="Got invalid task None"):
        llm_none._generate(["Hello"])

    # Test invalid string task
    mock_pipe_inv = MagicMock()
    mock_pipe_inv.task = "invalid-task"
    mock_pipe_inv.model.name_or_path = "mock-model-id"
    mock_pipe_inv.return_value = [{"text": "Something"}]

    llm_inv = HuggingFacePipeline(pipeline=mock_pipe_inv)
    with pytest.raises(ValueError, match="Got invalid task invalid-task"):
        llm_inv._generate(["Hello"])
