"""Test PromptLayer OpenAIChat API wrapper."""

from pathlib import Path

import pytest

from langchain_community.llms.loading import load_llm
from langchain_community.llms.promptlayer_openai import PromptLayerOpenAIChat


def test_promptlayer_openai_chat_call() -> None:
    """Test valid call to promptlayer openai."""
    llm = PromptLayerOpenAIChat(max_tokens=10)  # type: ignore[call-arg]
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


def test_promptlayer_openai_chat_stop_valid() -> None:
    """Test promptlayer openai stop logic on valid configuration."""
    query = "write an ordered list of five items"
    first_llm = PromptLayerOpenAIChat(stop="3", temperature=0)  # type: ignore[call-arg]
    first_output = first_llm.invoke(query)
    second_llm = PromptLayerOpenAIChat(temperature=0)  # type: ignore[call-arg]
    second_output = second_llm.invoke(query, stop=["3"])
    # Because it stops on new lines, shouldn't return anything
    assert first_output == second_output


def test_promptlayer_openai_chat_stop_error() -> None:
    """Test promptlayer openai stop logic on bad configuration."""
    llm = PromptLayerOpenAIChat(stop="3", temperature=0)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        llm.invoke("write an ordered list of five items", stop=["\n"])


def test_saving_loading_llm(tmp_path: Path) -> None:
    """Test saving/loading an promptlayer OpenAPI LLM."""
    llm = PromptLayerOpenAIChat(max_tokens=10)  # type: ignore[call-arg]
    llm.save(file_path=tmp_path / "openai.yaml")
    loaded_llm = load_llm(tmp_path / "openai.yaml")
    assert loaded_llm == llm
