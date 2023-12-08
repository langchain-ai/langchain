# flake8: noqa
"""Test Llama.cpp wrapper."""
import os
from urllib.request import urlretrieve

from langchain.llms import GPT4All


def _download_model() -> str:
    """Download model."""
    model_url = "http://gpt4all.io/models/ggml-gpt4all-l13b-snoozy.bin"
    local_filename = model_url.split("/")[-1]

    if not os.path.exists(local_filename):
        urlretrieve(model_url, local_filename)

    return local_filename


def test_gpt4all_inference() -> None:
    """Test valid gpt4all inference."""
    model_path = _download_model()
    llm = GPT4All(model=model_path)
    output = llm("Say foo:")
    assert isinstance(output, str)
