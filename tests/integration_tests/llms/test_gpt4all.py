# flake8: noqa
"""Test Llama.cpp wrapper."""
import os
from urllib.request import urlretrieve

from langchain.llms import GPT4All


def _download_model() -> str:
    """Download model.
    From https://the-eye.eu/public/AI/models/nomic-ai/gpt4all/gpt4all-lora-quantized.bin,
    convert to new ggml format and return model path."""
    model_url = "https://the-eye.eu/public/AI/models/nomic-ai/gpt4all/gpt4all-lora-quantized.bin"
    tokenizer_url = "https://huggingface.co/decapoda-research/llama-7b-hf/resolve/main/tokenizer.model"
    conversion_script = "https://github.com/nomic-ai/pyllamacpp/blob/main/pyllamacpp/scripts/convert_gpt4all.py"
    local_filename = model_url.split("/")[-1]

    if not os.path.exists("convert_gpt4all.py"):
        urlretrieve(conversion_script, "convert_gpt4all.py")
    if not os.path.exists("tokenizer.model"):
        urlretrieve(tokenizer_url, "tokenizer.model")
    if not os.path.exists(local_filename):
        urlretrieve(model_url, local_filename)
        os.system(f"python convert_gpt4all.py.py . tokenizer.model")

    return local_filename


def test_gpt4all_inference() -> None:
    """Test valid gpt4all inference."""
    model_path = _download_model()
    llm = GPT4All(model=model_path)
    output = llm("Say foo:")
    assert isinstance(output, str)
