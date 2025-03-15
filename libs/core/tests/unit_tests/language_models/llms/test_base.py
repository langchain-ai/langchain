from collections.abc import AsyncIterator, Iterator
from typing import Any, Optional

import pytest

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseLLM, FakeListLLM, FakeStreamingListLLM
from langchain_core.language_models.fake import FakeListLLMError
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.tracers.context import collect_runs
from tests.unit_tests.fake.callbacks import (
    BaseFakeCallbackHandler,
    FakeAsyncCallbackHandler,
    FakeCallbackHandler,
)


def test_batch() -> None:
    llm = FakeListLLM(responses=["foo"] * 3)
    output = llm.batch(["foo", "bar", "foo"])
    assert output == ["foo"] * 3

    output = llm.batch(["foo", "bar", "foo"], config={"max_concurrency": 2})
    assert output == ["foo"] * 3


async def test_abatch() -> None:
    llm = FakeListLLM(responses=["foo"] * 3)
    output = await llm.abatch(["foo", "bar", "foo"])
    assert output == ["foo"] * 3

    output = await llm.abatch(["foo", "bar", "foo"], config={"max_concurrency": 2})
    assert output == ["foo"] * 3


def test_batch_size() -> None:
    llm = FakeListLLM(responses=["foo"] * 3)
    with collect_runs() as cb:
        llm.batch(["foo", "bar", "foo"], {"callbacks": [cb]})
        assert all((r.extra or {}).get("batch_size") == 3 for r in cb.traced_runs)
        assert len(cb.traced_runs) == 3
    llm = FakeListLLM(responses=["foo"])
    with collect_runs() as cb:
        llm.batch(["foo"], {"callbacks": [cb]})
        assert all((r.extra or {}).get("batch_size") == 1 for r in cb.traced_runs)
        assert len(cb.traced_runs) == 1

    llm = FakeListLLM(responses=["foo"])
    with collect_runs() as cb:
        llm.invoke("foo")
        assert len(cb.traced_runs) == 1
        assert (cb.traced_runs[0].extra or {}).get("batch_size") == 1

    llm = FakeListLLM(responses=["foo"])
    with collect_runs() as cb:
        list(llm.stream("foo"))
        assert len(cb.traced_runs) == 1
        assert (cb.traced_runs[0].extra or {}).get("batch_size") == 1

    llm = FakeListLLM(responses=["foo"] * 1)
    with collect_runs() as cb:
        llm.invoke("foo")
        assert len(cb.traced_runs) == 1
        assert (cb.traced_runs[0].extra or {}).get("batch_size") == 1


async def test_async_batch_size() -> None:
    llm = FakeListLLM(responses=["foo"] * 3)
    with collect_runs() as cb:
        await llm.abatch(["foo", "bar", "foo"], {"callbacks": [cb]})
        assert all((r.extra or {}).get("batch_size") == 3 for r in cb.traced_runs)
        assert len(cb.traced_runs) == 3
    llm = FakeListLLM(responses=["foo"])
    with collect_runs() as cb:
        await llm.abatch(["foo"], {"callbacks": [cb]})
        assert all((r.extra or {}).get("batch_size") == 1 for r in cb.traced_runs)
        assert len(cb.traced_runs) == 1

    llm = FakeListLLM(responses=["foo"])
    with collect_runs() as cb:
        await llm.ainvoke("foo")
        assert len(cb.traced_runs) == 1
        assert (cb.traced_runs[0].extra or {}).get("batch_size") == 1

    llm = FakeListLLM(responses=["foo"])
    with collect_runs() as cb:
        async for _ in llm.astream("foo"):
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

    for i in range(2):
        llm = FakeStreamingListLLM(
            responses=[message],
            error_on_chunk_number=i,
        )
        with pytest.raises(FakeListLLMError):
            cb_async = FakeAsyncCallbackHandler()
            async for _ in llm.astream("Dummy message", callbacks=[cb_async]):
                pass
            eval_response(cb_async, i)

            cb_sync = FakeCallbackHandler()
            for _ in llm.stream("Dumy message", callbacks=[cb_sync]):
                pass

            eval_response(cb_sync, i)


async def test_astream_fallback_to_ainvoke() -> None:
    """Test astream uses appropriate implementation."""

    class ModelWithGenerate(BaseLLM):
        def _generate(
            self,
            prompts: list[str],
            stop: Optional[list[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> LLMResult:
            generations = [Generation(text="hello")]
            return LLMResult(generations=[generations])

        @property
        def _llm_type(self) -> str:
            return "fake-chat-model"

    model = ModelWithGenerate()
    chunks = list(model.stream("anything"))
    assert chunks == ["hello"]

    chunks = [chunk async for chunk in model.astream("anything")]
    assert chunks == ["hello"]


async def test_astream_implementation_fallback_to_stream() -> None:
    """Test astream uses appropriate implementation."""

    class ModelWithSyncStream(BaseLLM):
        def _generate(
            self,
            prompts: list[str],
            stop: Optional[list[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> LLMResult:
            """Top Level call."""
            raise NotImplementedError

        def _stream(
            self,
            prompt: str,
            stop: Optional[list[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> Iterator[GenerationChunk]:
            """Stream the output of the model."""
            yield GenerationChunk(text="a")
            yield GenerationChunk(text="b")

        @property
        def _llm_type(self) -> str:
            return "fake-chat-model"

    model = ModelWithSyncStream()
    chunks = list(model.stream("anything"))
    assert chunks == ["a", "b"]
    assert type(model)._astream == BaseLLM._astream
    astream_chunks = [chunk async for chunk in model.astream("anything")]
    assert astream_chunks == ["a", "b"]


async def test_astream_implementation_uses_astream() -> None:
    """Test astream uses appropriate implementation."""

    class ModelWithAsyncStream(BaseLLM):
        def _generate(
            self,
            prompts: list[str],
            stop: Optional[list[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> LLMResult:
            """Top Level call."""
            raise NotImplementedError

        async def _astream(
            self,
            prompt: str,
            stop: Optional[list[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> AsyncIterator[GenerationChunk]:
            """Stream the output of the model."""
            yield GenerationChunk(text="a")
            yield GenerationChunk(text="b")

        @property
        def _llm_type(self) -> str:
            return "fake-chat-model"

    model = ModelWithAsyncStream()
    chunks = [chunk async for chunk in model.astream("anything")]
    assert chunks == ["a", "b"]
