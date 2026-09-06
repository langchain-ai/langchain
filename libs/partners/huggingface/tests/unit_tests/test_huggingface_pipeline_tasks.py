"""Unit tests for HuggingFacePipeline task detection."""

from unittest.mock import MagicMock

from langchain_huggingface.llms.huggingface_pipeline import (
    HuggingFacePipeline,
    _is_supported_task,
    _is_translation_task,
)


def test_is_translation_task() -> None:
    assert _is_translation_task("translation")
    assert _is_translation_task("translation_en_to_fr")
    assert not _is_translation_task("summarization")
    assert not _is_translation_task("text-generation")


def test_is_supported_task_accepts_language_pair_translation() -> None:
    assert _is_supported_task("translation_en_to_fr")
    assert _is_supported_task("text-generation")
    assert not _is_supported_task("feature-extraction")


def test_generate_accepts_translation_xx_to_yy_task() -> None:
    pipe = MagicMock()
    pipe.task = "translation_en_to_fr"
    pipe.model.name_or_path = "mock-model-id"
    pipe.return_value = [{"translation_text": "Bonjour le monde"}]

    llm = HuggingFacePipeline(pipeline=pipe)
    result = llm._generate(["Hello world"])
    assert result.generations[0][0].text == "Bonjour le monde"
