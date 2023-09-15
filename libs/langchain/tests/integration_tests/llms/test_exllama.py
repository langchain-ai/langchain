# flake8: noqa
"""Test Llama.cpp wrapper."""
import os
from urllib.request import urlretrieve

import pytest

from langchain.llms import Exllama

from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


def get_model() -> str:
    """Download model.
    From https://huggingface.co/iambestfeed/open_llama_3b_4bit_128g,
    convert to new ggml format and return model path."""
    username = "iambestfeed"
    repo_name = "open_llama_3b_4bit_128g"
    model_url = f"https://huggingface.co/{username}/{repo_name}/resolve/main"
    model_files = [
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.model",
        "tokenizer_config.json",
    ]
    if not os.path.exists(f"/tmp/{repo_name}"):
        os.mkdir(f"/tmp/{repo_name}")
    for file in model_files:
        local_filename = f"/tmp/{repo_name}/{file}"
        if not os.path.exists(local_filename):
            urlretrieve(f"{model_url}/{file}", local_filename)
    return f"/tmp/{repo_name}"


def test_llamacpp_inference() -> None:
    """Test valid llama.cpp inference."""
    model_path = get_model()
    llm = Exllama(model_path=model_path)
    output = llm("Say foo:")
    assert isinstance(output, str)
    assert len(output) > 1
