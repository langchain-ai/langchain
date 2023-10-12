"""Test LLM callbacks."""
from langchain.chat_models.fake import FakeListChatModel
from langchain.llms.fake import FakeListLLM
from langchain.schema.messages import HumanMessage
from tests.unit_tests.callbacks.fake_callback_handler import (
    FakeCallbackHandler,
    FakeCallbackHandlerWithChatStart,
)


def test_llm_with_callbacks() -> None:
    """Test LLM callbacks."""
    handler = FakeCallbackHandler()
    llm = FakeListLLM(callbacks=[handler], verbose=True, responses=["foo"])
    output = llm("foo")
    assert output == "foo"
    assert handler.starts == 1
    assert handler.ends == 1
    assert handler.errors == 0


def test_chat_model_with_v1_callbacks() -> None:
    """Test chat model callbacks fall back to on_llm_start."""
    handler = FakeCallbackHandler()
    llm = FakeListChatModel(
        callbacks=[handler], verbose=True, responses=["fake response"]
    )
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
    llm = FakeListChatModel(
        callbacks=[handler], verbose=True, responses=["fake response"]
    )
    output = llm([HumanMessage(content="foo")])
    assert output.content == "fake response"
    assert handler.starts == 1
    assert handler.ends == 1
    assert handler.errors == 0
    assert handler.llm_starts == 0
    assert handler.llm_ends == 1
    assert handler.chat_model_starts == 1
