"""Test Replicate API wrapper."""

from langchain.callbacks.manager import CallbackManager
from langchain.llms.replicate import Replicate
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler

TEST_MODEL = "replicate/dolly-v2-12b:ef0e1aefc61f8e096ebe4db6b2bacc297daf2ef6899f0f7e001ec445893500e5"  # noqa: E501


def test_replicate_call() -> None:
    """Test simple non-streaming call to Replicate."""
    llm = Replicate(model=TEST_MODEL)
    output = llm("What is LangChain")
    assert output
    assert isinstance(output, str)


def test_replicate_streaming_call() -> None:
    """Test streaming call to Replicate."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])

    llm = Replicate(streaming=True, callback_manager=callback_manager, model=TEST_MODEL)
    output = llm("What is LangChain")
    assert output
    assert isinstance(output, str)


def test_replicate_model_kwargs() -> None:
    """Test simple non-streaming call to Replicate."""
    llm = Replicate(
        model=TEST_MODEL, model_kwargs={"max_length": 100, "temperature": 0.01}
    )
    long_output = llm("What is LangChain")
    llm = Replicate(
        model=TEST_MODEL, model_kwargs={"max_length": 10, "temperature": 0.01}
    )
    short_output = llm("What is LangChain")
    assert len(short_output) < len(long_output)
    assert llm.model_kwargs == {"max_length": 10, "temperature": 0.01}


def test_replicate_input() -> None:
    llm = Replicate(model=TEST_MODEL, input={"max_length": 10})
    assert llm.model_kwargs == {"max_length": 10}
