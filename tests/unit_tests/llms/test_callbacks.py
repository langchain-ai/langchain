"""Test LLM callbacks."""
from langchain.callbacks.base import CallbackManager
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler
from tests.unit_tests.llms.fake_llm import FakeLLM


def test_llm_with_callbacks() -> None:
    """Test LLM callbacks."""
    handler = FakeCallbackHandler()
    llm = FakeLLM(callback_manager=CallbackManager(handlers=[handler]), verbose=True)
    output = llm("foo")
    assert output == "foo"
    assert handler.starts == 1
    assert handler.ends == 1
    assert handler.errors == 0


def test_llm_with_callbacks_not_verbose() -> None:
    """Test LLM callbacks but not verbose."""
    import langchain

    langchain.verbose = False

    handler = FakeCallbackHandler()
    llm = FakeLLM(callback_manager=CallbackManager(handlers=[handler]))
    output = llm("foo")
    assert output == "foo"
    assert handler.starts == 0
    assert handler.ends == 0
    assert handler.errors == 0
