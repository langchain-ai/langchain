from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import PrivateAttr
from typing_extensions import override

from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    BaseCallbackHandler,
    CallbackManagerForChainRun,
)
from langchain_core.decision_models import (
    BaseDecisionModel,
    Choice,
    ChoiceAnswer,
    DecisionLevel,
    DecisionOption,
    DecisionRequest,
    DecisionResponse,
    DecisionResponseValidationError,
    DecisionUsage,
    FakeDecisionModel,
    Noul,
    NoulAnswer,
    RefusalAnswer,
    ScoreAnswer,
)
from langchain_core.decision_models import base as base_module
from langchain_core.exceptions import ModelInvalidRequestError
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.runnables.config import ensure_config

if TYPE_CHECKING:
    from uuid import UUID


class Recorder(BaseCallbackHandler):
    def __init__(self) -> None:
        self.starts: list[dict[str, Any]] = []
        self.ends = 0
        self.errors = 0

    @override
    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self.starts.append({"run_id": run_id, **kwargs})

    @override
    def on_chain_end(self, outputs: Any, **kwargs: Any) -> None:
        self.ends += 1

    @override
    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        self.errors += 1


def test_mixed_questions(
    decision_request: DecisionRequest, response: DecisionResponse
) -> None:
    recorder = Recorder()
    result = FakeDecisionModel(response=response).invoke(
        decision_request,
        {"callbacks": [recorder], "metadata": {"caller": "test"}, "tags": ["test"]},
    )
    assert result == response
    assert len(recorder.starts) == recorder.ends == 1
    assert recorder.errors == 0
    assert recorder.starts[0]["metadata"]["caller"] == "test"
    assert recorder.starts[0]["metadata"]["decision_model"] is True
    assert recorder.starts[0]["tags"] == ["test"]
    assert recorder.starts[0]["run_type"] == "llm"


def test_dict_questions(
    decision_request: DecisionRequest, response: DecisionResponse
) -> None:
    raw = {
        "state": decision_request["state"],
        "questions": {
            name: question.model_dump()
            for name, question in decision_request["questions"].items()
        },
    }
    model = FakeDecisionModel(response=response)
    assert model.invoke(raw) == response  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "state",
    [
        HumanMessage(content="hello"),
        {"nested": [HumanMessage(content="hello")]},
        ["hello", {"n": 2}],
    ],
)
def test_supported_state(
    state: Any, decision_request: DecisionRequest, response: DecisionResponse
) -> None:
    decision_request["state"] = state
    assert FakeDecisionModel(response=response).invoke(decision_request) == response


@pytest.mark.parametrize(
    "invalid_request",
    [
        {"state": "hello", "questions": {}},
        {"state": "hello", "questions": {" ": {"type": "noul", "instructions": "Hi?"}}},
        {
            "state": "hello",
            "questions": {"q": {"type": "predicate", "instructions": "Hi?"}},
        },
        {"state": {"n": float("nan")}, "questions": {"q": Noul(instructions="Hi?")}},
        {"state": {1: "invalid key"}, "questions": {"q": Noul(instructions="Hi?")}},
        {"state": object(), "questions": {"q": Noul(instructions="Hi?")}},
        {"state": "hello", "questions": {"q": Noul(instructions="Hi?")}, "extra": True},
    ],
)
def test_invalid_requests_have_error_lifecycle(
    invalid_request: Any, response: DecisionResponse
) -> None:
    recorder = Recorder()
    with pytest.raises(ModelInvalidRequestError):
        FakeDecisionModel(response=response).invoke(
            invalid_request, {"callbacks": [recorder]}
        )
    assert len(recorder.starts) == recorder.errors == 1
    assert recorder.ends == 0


def test_kwargs_rejected(
    decision_request: DecisionRequest, response: DecisionResponse
) -> None:
    with pytest.raises(ModelInvalidRequestError, match="keyword"):
        FakeDecisionModel(response=response).invoke(decision_request, temperature=0)


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "extra",
        "wrong_type",
        "probability",
        "mass",
        "choice",
        "duplicates",
        "score",
        "levels",
        "usage",
    ],
)
def test_invalid_responses(
    kind: str, decision_request: DecisionRequest, response: DecisionResponse
) -> None:
    model = FakeDecisionModel(response=response)
    answers = model.response.answers
    if kind == "missing":
        del answers["urgent"]
    elif kind == "extra":
        answers["extra"] = NoulAnswer(probability=0.5)
    elif kind == "wrong_type":
        answers["severity"] = NoulAnswer(probability=0.5)
    elif kind == "probability":
        answers["urgent"] = NoulAnswer.model_construct(probability=True)
    elif kind in {"mass", "choice", "duplicates"}:
        answer = answers["team"]
        assert isinstance(answer, ChoiceAnswer)
        if kind == "mass":
            answer.probabilities[0].probability = 0.5
        elif kind == "choice":
            answer.value = "invented"
        else:
            answer.probabilities[0].value = "technical"
    elif kind in {"score", "levels"}:
        answer = answers["severity"]
        assert isinstance(answer, ScoreAnswer)
        if kind == "score":
            answer.score = 0
        else:
            answer.levels[0] = DecisionLevel(label="invented")
    else:
        model.response.usage.input_tokens = True
    recorder = Recorder()
    with pytest.raises(DecisionResponseValidationError) as error:
        model.invoke(decision_request, {"callbacks": [recorder]})
    assert not error.value.is_retryable
    assert str(decision_request["state"]) not in str(error.value)
    assert len(recorder.starts) == recorder.errors == 1
    assert recorder.ends == 0


@pytest.mark.parametrize(
    "profile",
    [
        {"question_types": ["noul"]},
        {"state_types": ["json"]},
        {"max_questions": 2},
        {"max_options": 2},
    ],
)
def test_capability_checks(
    profile: Any, decision_request: DecisionRequest, response: DecisionResponse
) -> None:
    with pytest.raises(ModelInvalidRequestError):
        FakeDecisionModel(response=response, profile=profile).invoke(decision_request)


def test_boolean_choice_capability(response: DecisionResponse) -> None:
    decision_request: DecisionRequest = {
        "state": "hello",
        "questions": {
            "q": Choice(
                instructions="Which?",
                options=[DecisionOption(value=True), DecisionOption(value="true")],
            )
        },
    }
    with pytest.raises(ModelInvalidRequestError, match="boolean"):
        FakeDecisionModel(response=response, profile={"boolean_choices": False}).invoke(
            decision_request
        )


def test_refusal_is_a_result(
    decision_request: DecisionRequest, response: DecisionResponse
) -> None:
    response.answers["urgent"] = RefusalAnswer(response_metadata={"reason": "policy"})
    model = FakeDecisionModel(response=response)
    assert model.invoke(decision_request).answers["urgent"].type == "refusal"


def test_provider_confidence_and_abstention(
    decision_request: DecisionRequest, response: DecisionResponse
) -> None:
    answer = response.answers["team"]
    assert isinstance(answer, ChoiceAnswer)
    answer.provider_confidence = 0.12
    answer.abstained = True
    answer.response_metadata = {"confidence_method": "entropy"}
    result = (
        FakeDecisionModel(response=response).invoke(decision_request).answers["team"]
    )
    assert isinstance(result, ChoiceAnswer)
    assert result.provider_confidence == 0.12
    assert result.selected_probability == 0.9
    assert result.abstained is True
    assert result.response_metadata["confidence_method"] == "entropy"


def test_batch_stream_and_non_subclass(
    decision_request: DecisionRequest, response: DecisionResponse
) -> None:
    model = FakeDecisionModel(response=response)
    assert model.batch([decision_request, decision_request]) == [response, response]
    assert list(model.stream(decision_request)) == [response]
    compatible: RunnableLambda[DecisionRequest, DecisionResponse] = RunnableLambda(
        lambda _: response
    )
    project = RunnableLambda(lambda result: result.answers["urgent"].probability)
    assert (model | project).invoke(decision_request) == 0.95
    assert (compatible | project).invoke(decision_request) == 0.95
    result = model.invoke(decision_request)
    result.answers.clear()
    assert model.invoke(decision_request) == response


def test_batch_return_exceptions_and_fallback(
    decision_request: DecisionRequest, response: DecisionResponse
) -> None:
    invalid = FakeDecisionModel(response=DecisionResponse(answers={}))
    results = invalid.batch(
        [decision_request, decision_request], return_exceptions=True
    )
    assert all(
        isinstance(result, DecisionResponseValidationError) for result in results
    )
    fallback = FakeDecisionModel(response=response)
    assert invalid.with_fallbacks([fallback]).invoke(decision_request) == response


async def test_async_batch_stream_events(
    decision_request: DecisionRequest, response: DecisionResponse
) -> None:
    model = FakeDecisionModel(response=response)
    assert await model.ainvoke(decision_request) == response
    assert await model.abatch([decision_request, decision_request]) == [
        response,
        response,
    ]
    assert [item async for item in model.astream(decision_request)] == [response]
    events = [
        event async for event in model.astream_events(decision_request, version="v2")
    ]
    assert [event["event"] for event in events] == [
        "on_llm_start",
        "on_llm_end",
    ]


class SyncOnly(FakeDecisionModel):
    _seen: dict[str, Any] = PrivateAttr(default_factory=dict)

    _adecide = BaseDecisionModel._adecide

    @override
    def _decide(
        self,
        decision_request: DecisionRequest,
        *,
        config: RunnableConfig,
        run_manager: CallbackManagerForChainRun,
    ) -> DecisionResponse:
        self._seen.update(ensure_config()["metadata"])
        return self.response.model_copy(deep=True)


async def test_executor_preserves_context(
    decision_request: DecisionRequest, response: DecisionResponse
) -> None:
    model = SyncOnly(response=response)
    recorder = Recorder()
    assert (
        await model.ainvoke(
            decision_request,
            {"metadata": {"caller": "executor"}, "callbacks": [recorder]},
        )
        == response
    )
    assert model._seen["caller"] == "executor"
    assert len(recorder.starts) == recorder.ends == 1


async def test_async_error_lifecycle(decision_request: DecisionRequest) -> None:
    recorder = Recorder()
    with pytest.raises(DecisionResponseValidationError):
        await FakeDecisionModel(response=DecisionResponse(answers={})).ainvoke(
            decision_request, {"callbacks": [recorder]}
        )
    assert len(recorder.starts) == recorder.errors == 1
    assert recorder.ends == 0


def test_cyclic_state_rejected(response: DecisionResponse) -> None:
    state: dict[str, Any] = {}
    state["cycle"] = state
    with pytest.raises(ModelInvalidRequestError):
        FakeDecisionModel(response=response).invoke(
            {"state": state, "questions": {"q": Noul(instructions="Hi?")}}
        )


def test_parent_callback_ids(
    decision_request: DecisionRequest, response: DecisionResponse
) -> None:
    recorder = Recorder()
    prepare: RunnableLambda[DecisionRequest, DecisionRequest] = RunnableLambda(
        lambda x: x
    )
    chain = prepare | FakeDecisionModel(response=response)
    assert chain.invoke(decision_request, {"callbacks": [recorder]}) == response
    assert len(recorder.starts) == recorder.ends == 3
    root = recorder.starts[0]["run_id"]
    assert all(start["parent_run_id"] == root for start in recorder.starts[1:])


def test_reported_usage_survives_invalid_answers(
    decision_request: DecisionRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Run:
        def __init__(self) -> None:
            self.extra: dict[str, Any] = {}

    run = Run()
    monkeypatch.setattr(base_module, "get_current_run_tree", lambda: run)
    model = FakeDecisionModel(
        response=DecisionResponse(
            answers={},
            usage=DecisionUsage(input_tokens=42),
        )
    )
    with pytest.raises(DecisionResponseValidationError):
        model.invoke(decision_request)
    assert run.extra["metadata"]["usage_metadata"] == {"input_tokens": 42}


def test_tracing_failure_does_not_fail_decision(
    decision_request: DecisionRequest,
    response: DecisionResponse,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable() -> None:
        msg = "Tracing unavailable."
        raise RuntimeError(msg)

    monkeypatch.setattr(base_module, "get_current_run_tree", unavailable)
    assert FakeDecisionModel(response=response).invoke(decision_request) == response


async def test_native_async_cancellation(
    decision_request: DecisionRequest,
    response: DecisionResponse,
) -> None:
    started = asyncio.Event()
    blocked = asyncio.Event()

    class BlockingModel(FakeDecisionModel):
        @override
        async def _adecide(
            self,
            request: DecisionRequest,
            *,
            config: RunnableConfig,
            run_manager: AsyncCallbackManagerForChainRun,
        ) -> DecisionResponse:
            started.set()
            await blocked.wait()
            return self.response

    recorder = Recorder()
    task = asyncio.create_task(
        BlockingModel(response=response).ainvoke(
            decision_request, {"callbacks": [recorder]}
        )
    )
    await asyncio.wait_for(started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(recorder.starts) == recorder.errors == 1
    assert recorder.ends == 0
