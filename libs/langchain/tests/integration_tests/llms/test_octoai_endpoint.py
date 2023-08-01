"""Test OctoAI API wrapper."""

from pathlib import Path

import pytest

from langchain.llms.loading import load_llm
from langchain.llms.octoai_endpoint import OctoAIEndpoint
from tests.integration_tests.llms.utils import assert_llm_equality


def test_octoai_endpoint_text_generation() -> None:
    """Test valid call to OctoAI text generation model."""
    llm = OctoAIEndpoint(
        endpoint_url="https://mpt-7b-demo-kk0powt97tmb.octoai.cloud/generate",
        octoai_api_token="<octoai_api_token>",
        model_kwargs={
            "max_new_tokens": 200,
            "temperature": 0.75,
            "top_p": 0.95,
            "repetition_penalty": 1,
            "seed": None,
            "stop": [],
        },
    )

    output = llm("Which state is Los Angeles in?")
    print(output)
    assert isinstance(output, str)


def test_octoai_endpoint_call_error() -> None:
    """Test valid call to OctoAI that errors."""
    llm = OctoAIEndpoint(
        endpoint_url="https://mpt-7b-demo-kk0powt97tmb.octoai.cloud/generate",
        model_kwargs={"max_new_tokens": -1},
    )
    with pytest.raises(ValueError):
        llm("Which state is Los Angeles in?")


def test_saving_loading_endpoint_llm(tmp_path: Path) -> None:
    """Test saving/loading an OctoAIHub LLM."""
    llm = OctoAIEndpoint(
        endpoint_url="https://mpt-7b-demo-kk0powt97tmb.octoai.cloud/generate",
        octoai_api_token="<octoai_api_token>",
        model_kwargs={
            "max_new_tokens": 200,
            "temperature": 0.75,
            "top_p": 0.95,
            "repetition_penalty": 1,
            "seed": None,
            "stop": [],
        },
    )
    llm.save(file_path=tmp_path / "octoai.yaml")
    loaded_llm = load_llm(tmp_path / "octoai.yaml")
    assert_llm_equality(llm, loaded_llm)
