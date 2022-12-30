"""Test LLM callbacks."""
from langchain.callbacks.base import CallbackManager
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler
from tests.unit_tests.llms.fake_llm import FakeLLM


def test_llm_with_callbacks() -> None:
    """Test LLM callbacks."""
    handler = FakeCallbackHandler()
    llm = FakeLLM(callback_manager=CallbackManager([handler]))
    output = llm("foo")
    assert output == "foo"
    assert handler.starts == 1
    assert handler.ends == 1
    assert handler.errors == 0
