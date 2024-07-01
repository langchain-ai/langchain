"""Test IPEX LLM"""

import os
from typing import Any

import pytest
from langchain_core.outputs import LLMResult

from langchain_community.llms import IpexLLM

model_ids_to_test = os.getenv("TEST_IPEXLLM_MODEL_IDS") or ""
skip_if_no_model_ids = pytest.mark.skipif(
    not model_ids_to_test, reason="TEST_IPEXLLM_MODEL_IDS environment variable not set."
)
model_ids_to_test = [model_id.strip() for model_id in model_ids_to_test.split(",")]  # type: ignore


def load_model(model_id: str) -> Any:
    llm = IpexLLM.from_model_id(
        model_id=model_id,
        model_kwargs={"temperature": 0, "max_length": 16, "trust_remote_code": True},
    )
    return llm


def load_model_more_types(model_id: str, load_in_low_bit: str) -> Any:
    llm = IpexLLM.from_model_id(
        model_id=model_id,
        load_in_low_bit=load_in_low_bit,
        model_kwargs={"temperature": 0, "max_length": 16, "trust_remote_code": True},
    )
    return llm


@skip_if_no_model_ids
@pytest.mark.parametrize(
    "model_id",
    model_ids_to_test,
)
def test_call(model_id: str) -> None:
    """Test valid call."""
    llm = load_model(model_id)
    output = llm.invoke("Hello!")
    assert isinstance(output, str)


@skip_if_no_model_ids
@pytest.mark.parametrize(
    "model_id",
    model_ids_to_test,
)
def test_asym_int4(model_id: str) -> None:
    """Test asym int4 data type."""
    llm = load_model_more_types(model_id=model_id, load_in_low_bit="asym_int4")
    output = llm.invoke("Hello!")
    assert isinstance(output, str)


@skip_if_no_model_ids
@pytest.mark.parametrize(
    "model_id",
    model_ids_to_test,
)
def test_generate(model_id: str) -> None:
    """Test valid generate."""
    llm = load_model(model_id)
    output = llm.generate(["Hello!"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)


@skip_if_no_model_ids
@pytest.mark.parametrize(
    "model_id",
    model_ids_to_test,
)
def test_save_load_lowbit(model_id: str) -> None:
    """Test save and load lowbit model."""
    saved_lowbit_path = "/tmp/saved_model"
    llm = load_model(model_id)
    llm.model.save_low_bit(saved_lowbit_path)
    del llm
    loaded_llm = IpexLLM.from_model_id_low_bit(
        model_id=saved_lowbit_path,
        tokenizer_id=model_id,
        model_kwargs={"temperature": 0, "max_length": 16, "trust_remote_code": True},
    )
    output = loaded_llm.invoke("Hello!")
    assert isinstance(output, str)
