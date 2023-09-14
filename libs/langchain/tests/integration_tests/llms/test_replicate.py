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
