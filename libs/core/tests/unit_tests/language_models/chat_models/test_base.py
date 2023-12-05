"""Test base chat model."""

import pytest

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.outputs.llm_result import LLMResult
from langchain_core.tracers.context import collect_runs
from tests.unit_tests.fake.callbacks import (
    BaseFakeCallbackHandler,
    FakeAsyncCallbackHandler,
    FakeCallbackHandler,
)
from tests.unit_tests.fake.chat_model import FakeListChatModel


@pytest.fixture
def messages() -> list:
    return [
        SystemMessage(content="You are a test user."),
        HumanMessage(content="Hello, I am a test user."),
    ]


@pytest.fixture
def messages_2() -> list:
    return [
        SystemMessage(content="You are a test user."),
        HumanMessage(content="Hello, I not a test user."),
    ]


def test_batch_size(messages: list, messages_2: list) -> None:
    # The base endpoint doesn't support native batching,
    # so we expect batch_size to always be 1
    llm = FakeListChatModel(responses=[str(i) for i in range(100)])
    with collect_runs() as cb:
        llm.batch([messages, messages_2], {"callbacks": [cb]})
        assert len(cb.traced_runs) == 2
        assert all([(r.extra or {}).get("batch_size") == 1 for r in cb.traced_runs])
    with collect_runs() as cb:
        llm.batch([messages], {"callbacks": [cb]})
        assert all([(r.extra or {}).get("batch_size") == 1 for r in cb.traced_runs])
        assert len(cb.traced_runs) == 1

    with collect_runs() as cb:
        llm.invoke(messages)
        assert len(cb.traced_runs) == 1
        assert (cb.traced_runs[0].extra or {}).get("batch_size") == 1

    with collect_runs() as cb:
        list(llm.stream(messages))
        assert len(cb.traced_runs) == 1
        assert (cb.traced_runs[0].extra or {}).get("batch_size") == 1


async def test_async_batch_size(messages: list, messages_2: list) -> None:
    llm = FakeListChatModel(responses=[str(i) for i in range(100)])
    # The base endpoint doesn't support native batching,
    # so we expect batch_size to always be 1
    with collect_runs() as cb:
        await llm.abatch([messages, messages_2], {"callbacks": [cb]})
        assert all([(r.extra or {}).get("batch_size") == 1 for r in cb.traced_runs])
        assert len(cb.traced_runs) == 2
    with collect_runs() as cb:
        await llm.abatch([messages], {"callbacks": [cb]})
        assert all([(r.extra or {}).get("batch_size") == 1 for r in cb.traced_runs])
        assert len(cb.traced_runs) == 1

    with collect_runs() as cb:
        await llm.ainvoke(messages)
        assert len(cb.traced_runs) == 1
        assert (cb.traced_runs[0].extra or {}).get("batch_size") == 1

    with collect_runs() as cb:
        async for _ in llm.astream(messages):
            pass
        assert len(cb.traced_runs) == 1
        assert (cb.traced_runs[0].extra or {}).get("batch_size") == 1


async def test_stream_error_callback() -> None:
    message = "test"

    def eval_response(callback: BaseFakeCallbackHandler, i: int) -> None:
        assert callback.errors == 1
        assert len(callback.errors_args) == 1
        llm_result: LLMResult = callback.errors_args[0]["kwargs"]["response"]
        if i == 0:
            assert llm_result.generations == []
        else:
            assert llm_result.generations[0][0].text == message[:i]

    for i in range(0, 2):
        llm = FakeListChatModel(
            responses=[message],
            error_on_chunk_number=i,
        )
        with pytest.raises(Exception):
            cb_async = FakeAsyncCallbackHandler()
            async for _ in llm.astream("Dummy message", callbacks=[cb_async]):
                pass
            eval_response(cb_async, i)

            cb_sync = FakeCallbackHandler()
            for _ in llm.stream("Dumy message", callbacks=[cb_sync]):
                pass

            eval_response(cb_sync, i)
