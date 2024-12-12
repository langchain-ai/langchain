"""Test Replicate API wrapper."""

import os

from langchain_community.llms.replicate import Replicate
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler

TEST_MODEL_HELLO = (
    "replicate/hello-world:"
    + "5c7d5dc6dd8bf75c1acaa8565735e7986bc5b66206b55cca93cb72c9bf15ccaa"
)
TEST_MODEL_LANG = "meta/meta-llama-3-8b-instruct"


def test_replicate_call() -> None:
    """Test simple non-streaming call to Replicate."""
    llm = Replicate(model=TEST_MODEL_HELLO)
    output = llm.invoke("What is LangChain")
    assert output
    assert isinstance(output, str)


def test_replicate_streaming_call() -> None:
    """Test streaming call to Replicate."""
    callback_handler = FakeCallbackHandler()

    llm = Replicate(
        streaming=True, callbacks=[callback_handler], model=TEST_MODEL_HELLO
    )
    output = llm.invoke("What is LangChain")
    assert output
    assert isinstance(output, str)


def test_replicate_model_kwargs() -> None:
    """Test simple non-streaming call to Replicate."""
    llm = Replicate(  # type: ignore[call-arg]
        model=TEST_MODEL_LANG, model_kwargs={"max_new_tokens": 10, "temperature": 0.01}
    )
    long_output = llm.invoke("What is LangChain")
    llm = Replicate(  # type: ignore[call-arg]
        model=TEST_MODEL_LANG, model_kwargs={"max_new_tokens": 5, "temperature": 0.01}
    )
    short_output = llm.invoke("What is LangChain")
    assert len(short_output) < len(long_output)
    assert llm.model_kwargs == {"max_new_tokens": 5, "temperature": 0.01}


def test_replicate_input() -> None:
    llm = Replicate(model=TEST_MODEL_LANG, input={"max_new_tokens": 10})
    assert llm.model_kwargs == {"max_new_tokens": 10}


def test_replicate_api_token_propagation() -> None:
    """Test that API token passed to the model is used to access the service."""
    # Grab the api token from the environment variable.
    api_token = os.getenv("REPLICATE_API_TOKEN")

    # Reset the environment variable to ensure it's not available.
    os.environ["REPLICATE_API_TOKEN"] = "yo"

    # Pass the api token into the model.
    llm = Replicate(model=TEST_MODEL_HELLO, replicate_api_token=api_token)
    output = llm.invoke("What is a duck?")

    assert output
    assert isinstance(output, str)
