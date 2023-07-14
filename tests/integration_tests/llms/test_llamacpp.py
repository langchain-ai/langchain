# flake8: noqa
"""Test Llama.cpp wrapper."""
import os
from typing import Generator
from urllib.request import urlretrieve

from langchain.llms import LlamaCpp

from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


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
    assert len(output) > 1


def test_llamacpp_streaming() -> None:
    """Test streaming tokens from LlamaCpp."""
    model_path = get_model()
    llm = LlamaCpp(model_path=model_path, max_tokens=10)
    generator = llm.stream("Q: How do you say 'hello' in German? A:'", stop=["'"])
    stream_results_string = ""
    assert isinstance(generator, Generator)

    for chunk in generator:
        assert not isinstance(chunk, str)
        # Note that this matches the OpenAI format:
        assert isinstance(chunk["choices"][0]["text"], str)
        stream_results_string += chunk["choices"][0]["text"]
    assert len(stream_results_string.strip()) > 1


def test_llamacpp_streaming_callback() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    MAX_TOKENS = 5
    OFF_BY_ONE = 1  # There may be an off by one error in the upstream code!

    callback_handler = FakeCallbackHandler()
    llm = LlamaCpp(
        model_path=get_model(),
        callbacks=[callback_handler],
        verbose=True,
        max_tokens=MAX_TOKENS,
    )
    llm("Q: Can you count to 10? A:'1, ")
    assert callback_handler.llm_streams <= MAX_TOKENS + OFF_BY_ONE
