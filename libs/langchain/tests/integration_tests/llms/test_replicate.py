"""Test Replicate API wrapper."""

from langchain.callbacks.manager import CallbackManager
from langchain.llms.replicate import Replicate
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler

TEST_MODEL_NAME = "replicate/hello-world"
TEST_MODEL_VER = "5c7d5dc6dd8bf75c1acaa8565735e7986bc5b66206b55cca93cb72c9bf15ccaa"
TEST_MODEL = TEST_MODEL_NAME + ":" + TEST_MODEL_VER


def test_replicate_call() -> None:
    """Test simple non-streaming call to Replicate."""
    llm = Replicate(model=TEST_MODEL)
    output = llm("LangChain")
    assert output == "hello LangChain"


def test_replicate_streaming_call() -> None:
    """Test streaming call to Replicate."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])

    llm = Replicate(streaming=True, callback_manager=callback_manager, model=TEST_MODEL)
    output = llm("LangChain")
    assert output == "hello LangChain"
    assert callback_handler.llm_streams == 15


def test_replicate_stop_sequence() -> None:
    """Test call to Replicate with a stop sequence."""
    llm = Replicate(model=TEST_MODEL)
    output = llm("one two three", stop=["two"])
    assert output == "hello one "
