"""Test HuggingFace text-generation API wrapper."""
import os

import pytest

from langchain.callbacks.manager import CallbackManager
from langchain.llms.huggingface_textgen import HuggingFaceTextgen
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


def test_huggingface_textgen_call() -> None:
    """Test valid call to HuggingFace text generation model."""
    if "HF_TEXTGEN_HOST" in os.environ and "HF_TEXTGEN_PORT" in os.environ:
        llm = HuggingFaceTextgen(
            host=os.environ["HF_TEXTGEN_HOST"],
            port=int(os.environ["HF_TEXTGEN_PORT"]),
            max_new_tokens=10,
            verbose=True,
        )
        output = llm("What is deep learning?")
        assert isinstance(output, str)
    else:
        pytest.skip(
            "Skipping test because HF_TEXTGEN_HOST and HF_TEXTGEN_PORT are not set."
        )


def test_huggingface_textgen_streaming_callback() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    if "HF_TEXTGEN_HOST" in os.environ and "HF_TEXTGEN_PORT" in os.environ:
        callback_handler = FakeCallbackHandler()
        callback_manager = CallbackManager([callback_handler])
        llm = HuggingFaceTextgen(
            host=os.environ["HF_TEXTGEN_HOST"],
            port=int(os.environ["HF_TEXTGEN_PORT"]),
            stream=True,
            callback_manager=callback_manager,
            max_new_tokens=10,
            verbose=True,
        )
        llm("What is deep learning?")
        assert callback_handler.llm_streams == 10
    else:
        pytest.skip(
            "Skipping test because HF_TEXTGEN_HOST and HF_TEXTGEN_PORT are not set."
        )
