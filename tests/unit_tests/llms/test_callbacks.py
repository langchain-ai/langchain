"""Test LLM callbacks."""
from langchain.schema import HumanMessage
from tests.unit_tests.callbacks.fake_callback_handler import (
    FakeCallbackHandler,
    FakeCallbackHandlerWithChatStart,
)
from tests.unit_tests.llms.fake_chat_model import FakeChatModel
from tests.unit_tests.llms.fake_llm import FakeLLM


def test_llm_with_callbacks() -> None:
    """Test LLM callbacks."""
    handler = FakeCallbackHandler()
    llm = FakeLLM(callbacks=[handler], verbose=True)
    output = llm("foo")
    assert output == "foo"
    assert handler.starts == 1
    assert handler.ends == 1
    assert handler.errors == 0


def test_chat_model_with_v1_callbacks() -> None:
    """Test chat model callbacks fall back to on_llm_start."""
    handler = FakeCallbackHandler()
    llm = FakeChatModel(callbacks=[handler], verbose=True)
    output = llm([HumanMessage(content="foo")])
    assert output.content == "fake response"
    assert handler.starts == 1
    assert handler.ends == 1
    assert handler.errors == 0
    assert handler.llm_starts == 1
    assert handler.llm_ends == 1


def test_chat_model_with_v2_callbacks() -> None:
    """Test chat model callbacks fall back to on_llm_start."""
    handler = FakeCallbackHandlerWithChatStart()
    llm = FakeChatModel(callbacks=[handler], verbose=True)
    output = llm([HumanMessage(content="foo")])
    assert output.content == "fake response"
    assert handler.starts == 1
    assert handler.ends == 1
    assert handler.errors == 0
    assert handler.llm_starts == 0
    assert handler.llm_ends == 1
    assert handler.chat_model_starts == 1
