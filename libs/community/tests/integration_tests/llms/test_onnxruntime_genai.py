# flake8: noqa
"""Test Llama.cpp wrapper."""

import os
from typing import Generator
from urllib.request import urlretrieve

import pytest

from langchain_community.llms.onnxruntime_genai import OnnxruntimeGenAi



def get_model_path():
    return "Local file_path"

def test_onnxruntime_genai_inference() -> None:
    """Test valid llama.cpp inference."""

    llm = OnnxruntimeGenAi(model_path=get_model_path())
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)
    assert len(output) > 1


def test_onnxruntime_genai_model_kwargs() -> None:
    llm = OnnxruntimeGenAi(model_path=get_model_path(), model_kwargs={"n_gqa": None})
    assert llm.model_kwargs == {"n_gqa": None}


def test_onnxruntime_genai_invalid_model_kwargs() -> None:
    with pytest.raises(ValueError):
        OnnxruntimeGenAi(model_path=get_model_path(), model_kwargs={"n_ctx": 1024})


def test_onnxruntime_genai_incorrect_field() -> None:
    with pytest.warns(match="not default parameter"):
        llm = OnnxruntimeGenAi(model_path=get_model_path(), n_gqa=None)
    llm.model_kwargs == {"n_gqa": None}
