import warnings
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any, get_type_hints

import pytest
from pydantic import Field
from typing_extensions import override

from langchain_core._api import LangChainDeprecationWarning
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    BaseCallbackHandler,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import (
    LLM,
    BaseLLM,
    FakeListLLM,
)
from langchain_core.language_models._utils import _filter_invocation_params_for_tracing
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.prompts import PromptTemplate
from langchain_core.tracers.context import collect_runs
from tests.unit_tests.fake.callbacks import (
    BaseFakeCallbackHandler,
    FakeAsyncCallbackHandler,
    FakeCallbackHandler,
)

if TYPE_CHECKING:
    from langchain_core.runnables.config import RunnableConfig


def test_asdict_replaces_deprecated_dict() -> None:
    llm = FakeListLLM(responses=["foo"])

    expected = {"responses": ["foo"], "_type": "fake-list"}
    assert llm.asdict() == expected
    with pytest.warns(LangChainDeprecationWarning, match="asdict"):
        assert llm.dict() == expected


def test_base_llm_type_hints_resolve() -> None:
    assert get_type_hints(BaseLLM.asdict)["return"] == dict[str, Any]


def test_invoke_preserves_deprecated_dict_override() -> None:
    """Invoking should preserve `dict()` overrides until `dict()` is removed."""

    class CustomDictLLM(FakeListLLM):
        @override
        def dict(self, **kwargs: Any) -> dict[str, Any]:
            data = super().dict(**kwargs)
            data["custom_trace_param"] = "custom"
            return data

    llm = CustomDictLLM(responses=["foo"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", LangChainDeprecationWarning)
        with collect_runs() as cb:
            assert llm.invoke("hello") == "foo"

    assert cb.traced_runs[0].extra is not None
    assert cb.traced_runs[0].extra["invocation_params"]["custom_trace_param"] == (
        "custom"
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


async def test_error_callback() -> None:
    class FailingLLMError(Exception):
        """FailingLLMError."""

    class FailingLLM(LLM):
        @property
        def _llm_type(self) -> str:
            """Return type of llm."""
            return "failing-llm"

        @override
        def _call(
            self,
            prompt: str,
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> str:
            raise FailingLLMError

    def eval_response(callback: BaseFakeCallbackHandler) -> None:
        assert callback.errors == 1
        assert len(callback.errors_args) == 1
        assert isinstance(callback.errors_args[0]["args"][0], FailingLLMError)

    llm = FailingLLM()
    cb_async = FakeAsyncCallbackHandler()
    with pytest.raises(FailingLLMError):
        await llm.ainvoke("Dummy message", config={"callbacks": [cb_async]})
    eval_response(cb_async)

    cb_sync = FakeCallbackHandler()
    with pytest.raises(FailingLLMError):
        llm.invoke("Dummy message", config={"callbacks": [cb_sync]})
    eval_response(cb_sync)


async def test_astream_fallback_to_ainvoke() -> None:
    """Test astream uses appropriate implementation."""

    class ModelWithGenerate(BaseLLM):
        @override
        def _generate(
            self,
            prompts: list[str],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
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
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> LLMResult:
            """Top Level call."""
            raise NotImplementedError

        @override
        def _stream(
            self,
            prompt: str,
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
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
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> LLMResult:
            """Top Level call."""
            raise NotImplementedError

        @override
        async def _astream(
            self,
            prompt: str,
            stop: list[str] | None = None,
            run_manager: AsyncCallbackManagerForLLMRun | None = None,
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


def test_get_ls_params() -> None:
    class LSParamsModel(BaseLLM):
        model: str = "foo"
        temperature: float = 0.1
        max_tokens: int = 1024

        @override
        def _generate(
            self,
            prompts: list[str],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> LLMResult:
            raise NotImplementedError

        @property
        def _llm_type(self) -> str:
            return "fake-model"

    llm = LSParamsModel()

    # Test standard tracing params
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "lsparamsmodel",
        "ls_model_type": "llm",
        "ls_model_name": "foo",
        "ls_temperature": 0.1,
        "ls_max_tokens": 1024,
    }

    ls_params = llm._get_ls_params(model="bar")
    assert ls_params["ls_model_name"] == "bar"

    ls_params = llm._get_ls_params(temperature=0.2)
    assert ls_params["ls_temperature"] == 0.2

    # Test integer temperature values (regression test for issue #35300)
    ls_params = llm._get_ls_params(temperature=0)
    assert ls_params["ls_temperature"] == 0

    ls_params = llm._get_ls_params(temperature=1)
    assert ls_params["ls_temperature"] == 1

    ls_params = llm._get_ls_params(max_tokens=2048)
    assert ls_params["ls_max_tokens"] == 2048

    ls_params = llm._get_ls_params(stop=["stop"])
    assert ls_params["ls_stop"] == ["stop"]


def test_filter_invocation_params_for_tracing() -> None:
    """Test that large fields are filtered from invocation params for tracing."""
    params = {
        "temperature": 0.7,
        "tools": [{"name": "test_tool"}],
        "functions": [{"name": "test_function"}],
        "messages": [{"role": "system", "content": "test"}],
        "response_format": {"type": "json_object"},
    }
    filtered = _filter_invocation_params_for_tracing(params)

    # Should include temperature
    assert "temperature" in filtered
    assert filtered["temperature"] == 0.7

    # Should exclude these large fields
    assert "tools" not in filtered
    assert "functions" not in filtered
    assert "messages" not in filtered
    assert "response_format" not in filtered


class FakeLLMWithInvocationParams(BaseLLM):
    """Fake LLM with invocation params for testing tracing."""

    temperature: float = 0.7

    @property
    @override
    def _llm_type(self) -> str:
        return "fake-llm-with-invocation-params"

    @property
    @override
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "tools": [{"name": "test_tool"}],
            "functions": [{"name": "test_function"}],
            "messages": [{"role": "system", "content": "test"}],
            "response_format": {"type": "json_object"},
        }

    @override
    def _generate(
        self,
        prompts: list[str],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations = [[Generation(text="test response")]]
        return LLMResult(generations=generations)

    @override
    async def _agenerate(
        self,
        prompts: list[str],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations = [[Generation(text="test response")]]
        return LLMResult(generations=generations)


async def test_llm_invocation_params_filtered_in_stream() -> None:
    """Test that invocation params are filtered when streaming."""

    # Create a custom LLM that supports streaming
    class FakeStreamingLLM(FakeLLMWithInvocationParams):
        @override
        def _stream(
            self,
            prompt: str,
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> Iterator[GenerationChunk]:
            yield GenerationChunk(text="test ")

    streaming_llm = FakeStreamingLLM()

    with collect_runs() as cb:
        list(streaming_llm.stream("Hello", config={"callbacks": [cb]}))
        assert len(cb.traced_runs) == 1
        run = cb.traced_runs[0]
        # Verify the run was traced
        assert run.extra is not None


class _KwargsRecordingLLM(LLM):
    """Records the keyword arguments that reach the provider methods."""

    provider_kwargs: list[dict[str, Any]] = Field(default_factory=list)

    @override
    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        self.provider_kwargs.append(kwargs)
        return "sunny"

    @override
    def _stream(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        self.provider_kwargs.append(kwargs)
        yield GenerationChunk(text="sunny")

    @property
    def _llm_type(self) -> str:
        return "kwargs-recording"


class _LLMStartRecorder(BaseCallbackHandler):
    def __init__(self) -> None:
        self.options: list[dict[str, Any]] = []

    @override
    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> None:
        self.options.append(kwargs["options"])


_CITY_PROMPT = PromptTemplate.from_template("Weather in {city}?")
_CITY_TEMPLATE_INFO = {
    "template": "Weather in {city}?",
    "template_format": "f-string",
    "variables": {"city": "Oslo"},
}


@pytest.mark.parametrize("method", ["invoke", "stream", "ainvoke", "astream"])
async def test_prompt_template_passed_to_llm_callbacks(method: str) -> None:
    llm = _KwargsRecordingLLM()
    recorder = _LLMStartRecorder()
    chain = _CITY_PROMPT | llm
    config: RunnableConfig = {"callbacks": [recorder]}

    if method == "invoke":
        chain.invoke({"city": "Oslo"}, config)
    elif method == "stream":
        list(chain.stream({"city": "Oslo"}, config))
    elif method == "ainvoke":
        await chain.ainvoke({"city": "Oslo"}, config)
    else:
        async for _ in chain.astream({"city": "Oslo"}, config):
            pass

    assert recorder.options == [
        {"stop": None, "ls_prompt_template": _CITY_TEMPLATE_INFO}
    ]
    assert llm.provider_kwargs
    assert all("ls_prompt_template" not in kw for kw in llm.provider_kwargs)


def test_no_prompt_template_for_plain_llm_input() -> None:
    recorder = _LLMStartRecorder()

    _KwargsRecordingLLM().invoke("Weather in Oslo?", {"callbacks": [recorder]})

    assert recorder.options == [{"stop": None}]
