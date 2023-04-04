# flake8: noqa
"""Test Llama.cpp wrapper."""
import os
from urllib.request import urlretrieve

from langchain.llms import LlamaCpp


def get_model() -> str:
    """Download model. f
    From https://huggingface.co/Sosaka/Alpaca-native-4bit-ggml/,
    convert to new ggml format and return model path."""
    model_url = "https://huggingface.co/Sosaka/Alpaca-native-4bit-ggml/resolve/main/ggml-alpaca-7b-q4.bin"
    tokenizer_url = "https://huggingface.co/decapoda-research/llama-7b-hf/resolve/main/tokenizer.model"
    conversion_script = "https://github.com/ggerganov/llama.cpp/raw/master/convert-unversioned-ggml-to-ggml.py"
    local_filename = model_url.split("/")[-1]

    if not os.path.exists("convert-unversioned-ggml-to-ggml.py"):
        urlretrieve(conversion_script, "convert-unversioned-ggml-to-ggml.py")
    if not os.path.exists("tokenizer.model"):
        urlretrieve(tokenizer_url, "tokenizer.model")
    if not os.path.exists(local_filename):
        urlretrieve(model_url, local_filename)
        os.system(f"python convert-unversioned-ggml-to-ggml.py . tokenizer.model")

    return local_filename


def test_llamacpp_inference() -> None:
    """Test valid llama.cpp inference."""
    model_path = get_model()
    llm = LlamaCpp(model_path=model_path)
    output = llm("Say foo:")
    assert isinstance(output, str)
