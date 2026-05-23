from unittest.mock import MagicMock, patch

import pytest

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline


@patch("langchain_huggingface.llms.huggingface_pipeline.is_optimum_intel_available")
@patch("langchain_huggingface.llms.huggingface_pipeline.is_optimum_intel_version")
@patch("langchain_huggingface.llms.huggingface_pipeline.is_ipex_available")
def test_huggingface_pipeline_ipex_unsupported_version(
    mock_ipex_available: MagicMock,
    mock_intel_version: MagicMock,
    mock_intel_available: MagicMock,
) -> None:
    """Test optimum-intel >= 2.0 raises ImportError for pipeline."""
    mock_intel_available.return_value = True
    mock_intel_version.return_value = True  # >= 2.0 evaluates to True
    mock_ipex_available.return_value = True

    with pytest.raises(ImportError) as exc_info:
        HuggingFacePipeline.from_model_id(
            model_id="gpt2",
            task="text-generation",
            backend="ipex",
        )
    assert "IPEX support has been removed in optimum-intel v2" in str(exc_info.value)
    assert "Please downgrade optimum-intel or switch backend." in str(exc_info.value)


@patch("langchain_huggingface.llms.huggingface_pipeline.is_optimum_intel_available")
@patch("langchain_huggingface.llms.huggingface_pipeline.is_optimum_intel_version")
@patch("langchain_huggingface.llms.huggingface_pipeline.is_ipex_available")
@patch("transformers.AutoTokenizer.from_pretrained")
@patch("transformers.pipeline")
def test_huggingface_pipeline_ipex_supported_version(
    mock_pipeline: MagicMock,
    mock_tokenizer: MagicMock,
    mock_ipex_available: MagicMock,
    mock_intel_version: MagicMock,
    mock_intel_available: MagicMock,
) -> None:
    """Test optimum-intel < 2.0 succeeds for pipeline."""
    mock_intel_available.return_value = True
    # For >= 2.0, return False (meaning the version is < 2.0)
    mock_intel_version.return_value = False
    mock_ipex_available.return_value = True

    mock_optimum_intel = MagicMock()
    modules = {"optimum": MagicMock(), "optimum.intel": mock_optimum_intel}
    with patch.dict("sys.modules", modules):
        mock_tokenizer.return_value = MagicMock(pad_token_id=0)
        mock_model = MagicMock()
        mock_model.config.pad_token_id = 0

        mock_from_pretrained = mock_optimum_intel.IPEXModelForCausalLM.from_pretrained
        mock_from_pretrained.return_value = mock_model

        mock_pipe = MagicMock()
        mock_pipe.task = "text-generation"
        mock_pipe.model = mock_model
        mock_pipeline.return_value = mock_pipe

        llm = HuggingFacePipeline.from_model_id(
            model_id="gpt2",
            task="text-generation",
            backend="ipex",
        )
        assert llm.model_id == "gpt2"
        mock_from_pretrained.assert_called_once()


@patch("langchain_huggingface.embeddings.huggingface.is_optimum_intel_available")
@patch("langchain_huggingface.embeddings.huggingface.is_optimum_intel_version")
@patch("langchain_huggingface.embeddings.huggingface.is_ipex_available")
def test_huggingface_embeddings_ipex_unsupported_version(
    mock_ipex_available: MagicMock,
    mock_intel_version: MagicMock,
    mock_intel_available: MagicMock,
) -> None:
    """Test optimum-intel >= 2.0 raises ImportError for embeddings."""
    mock_intel_available.return_value = True
    mock_intel_version.return_value = True  # >= 2.0 evaluates to True
    mock_ipex_available.return_value = True

    with pytest.raises(ImportError) as exc_info:
        HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"backend": "ipex"},
        )
    assert "IPEX support has been removed in optimum-intel v2" in str(exc_info.value)
    assert "Please downgrade optimum-intel or switch backend." in str(exc_info.value)


@patch("langchain_huggingface.embeddings.huggingface.is_optimum_intel_available")
@patch("langchain_huggingface.embeddings.huggingface.is_optimum_intel_version")
@patch("langchain_huggingface.embeddings.huggingface.is_ipex_available")
def test_huggingface_embeddings_ipex_supported_version(
    mock_ipex_available: MagicMock,
    mock_intel_version: MagicMock,
    mock_intel_available: MagicMock,
) -> None:
    """Test optimum-intel < 2.0 succeeds for embeddings."""
    mock_intel_available.return_value = True
    # For >= 2.0, return False (meaning the version is < 2.0)
    mock_intel_version.return_value = False
    mock_ipex_available.return_value = True

    mock_optimum_intel = MagicMock()
    modules = {"optimum": MagicMock(), "optimum.intel": mock_optimum_intel}
    with patch.dict("sys.modules", modules):
        mock_client = MagicMock()
        mock_optimum_intel.IPEXSentenceTransformer.return_value = mock_client

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"backend": "ipex"},
        )
        assert embeddings.model_name == "sentence-transformers/all-mpnet-base-v2"
        mock_optimum_intel.IPEXSentenceTransformer.assert_called_once()
