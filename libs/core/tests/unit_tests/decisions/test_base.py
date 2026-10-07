"""Unit tests for `BaseDecisionModel`."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import Field, ValidationError
from typing_extensions import override

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.decisions import (
    BaseDecisionModel,
    Choice,
    ChoiceAnswer,
    DecisionRequest,
    DecisionResponse,
    Level,
    Predicate,
    PredicateAnswer,
    RefusalAnswer,
    Score,
    ScoreAnswer,
)
from langchain_core.decisions import base as base_module

pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_core._api.LangChainBetaWarning"
)


def _response() -> DecisionResponse:
    return DecisionResponse(
        model="fake-model",
        answers={
            "urgent": PredicateAnswer(type="predicate", probability=0.9),
            "dept": ChoiceAnswer(
                type="choice",
                choice="billing",
                probabilities={"billing": 0.8, "technical": 0.2},
                confidence=0.6,
            ),
            "severity": ScoreAnswer(
                type="score",
                score=1.2,
                legend={0: "low", 1: "medium", 2: "high"},
                probabilities={0: 0.1, 1: 0.6, 2: 0.3},
                confidence=0.4,
            ),
            "refused": RefusalAnswer(type="refusal"),
        },
        usage={"input_tokens": 42, "output_tokens": 0},
    )


class _FakeDecisionModel(BaseDecisionModel):
    """Decision model that records requests and returns a fixed response."""

    requests: list[Any] = Field(default_factory=list)

    @property
    @override
    def _provider(self) -> str:
        return "fake"

    @override
    def _decide(self, request: DecisionRequest) -> DecisionResponse:
        self.requests.append(request)
        return _response()


class _AsyncFakeDecisionModel(_FakeDecisionModel):
    async_calls: int = 0

    @override
    async def _adecide(self, request: DecisionRequest) -> DecisionResponse:
        self.async_calls += 1
        return _response()


class _RunTreeStub:
    def __init__(self) -> None:
        self.extra: dict[str, Any] = {}


class _RunRecorder(BaseCallbackHandler):
    def __init__(self) -> None:
        self.metadata: dict[str, Any] = {}
        self.run_type: str | None = None
        self.ends = 0

    def on_chain_start(self, *_: Any, **kwargs: Any) -> None:
        self.metadata = kwargs.get("metadata") or {}
        self.run_type = kwargs.get("run_type")

    def on_chain_end(self, *_: Any, **__: Any) -> None:
        self.ends += 1


def _request() -> DecisionRequest:
    return {
        "input": "hello",
        "questions": {"urgent": Predicate(instructions="Is this urgent?")},
    }


def test_is_beta() -> None:
    assert (BaseDecisionModel.__doc__ or "").startswith(".. beta::")


def test_cannot_instantiate_without_provider_methods() -> None:
    with pytest.raises(TypeError, match="abstract"):
        BaseDecisionModel(model="x")  # type: ignore[abstract]


def test_model_is_required() -> None:
    with pytest.raises(ValidationError, match="model"):
        _FakeDecisionModel()  # type: ignore[call-arg]


def test_invoke_delegates_to_decide() -> None:
    model = _FakeDecisionModel(model="fake-model")

    response = model.invoke(_request())

    assert model.requests == [_request()]
    assert response == _response()


async def test_ainvoke_runs_decide_in_executor_by_default() -> None:
    model = _FakeDecisionModel(model="fake-model")

    response = await model.ainvoke(_request())

    assert model.requests == [_request()]
    assert response == _response()


async def test_ainvoke_uses_native_adecide() -> None:
    model = _AsyncFakeDecisionModel(model="fake-model")

    await model.ainvoke(_request())

    assert model.async_calls == 1
    assert model.requests == []


def test_run_carries_provider_identity_and_caller_metadata() -> None:
    recorder = _RunRecorder()

    _FakeDecisionModel(model="fake-model").invoke(
        _request(), config={"callbacks": [recorder], "metadata": {"tenant": "acme"}}
    )

    assert recorder.run_type == "llm"
    assert recorder.ends == 1
    assert recorder.metadata["tenant"] == "acme"
    assert recorder.metadata["ls_provider"] == "fake"
    assert recorder.metadata["ls_model_name"] == "fake-model"


@pytest.mark.parametrize("async_", [False, True])
async def test_usage_is_recorded_on_the_active_run(
    monkeypatch: pytest.MonkeyPatch, *, async_: bool
) -> None:
    stub = _RunTreeStub()
    monkeypatch.setattr(base_module, "get_current_run_tree", lambda: stub)
    model = _AsyncFakeDecisionModel(model="fake-model")

    if async_:
        await model.ainvoke(_request())
    else:
        model.invoke(_request())

    assert stub.extra["metadata"]["usage_metadata"] == {
        "input_tokens": 42,
        "output_tokens": 0,
        "total_tokens": 42,
    }


def test_tracing_failure_does_not_fail_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken() -> None:
        msg = "tracing unavailable"
        raise RuntimeError(msg)

    monkeypatch.setattr(base_module, "get_current_run_tree", broken)

    response = _FakeDecisionModel(model="fake-model").invoke(_request())

    assert response.usage.input_tokens == 42


def test_response_views_filter_by_answer_type() -> None:
    response = _response()

    assert set(response.predicates) == {"urgent"}
    assert set(response.choices) == {"dept"}
    assert set(response.scores) == {"severity"}
    assert set(response.refusals) == {"refused"}


def test_score_as_levels_expands_shorthand() -> None:
    score = Score(
        instructions="How severe?",
        levels=["low", Level(label="high", description="Blocked.")],
    )

    assert score.as_levels() == [
        Level(label="low"),
        Level(label="high", description="Blocked."),
    ]


@pytest.mark.parametrize(
    ("question_type", "kwargs"),
    [
        (Choice, {"instructions": "Which?", "choices": {}}),
        (Score, {"instructions": "How much?", "levels": ["only"]}),
        (Predicate, {"instructions": ""}),
    ],
)
def test_questions_validate_their_options(
    question_type: type[Predicate | Choice | Score], kwargs: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError):
        question_type(**kwargs)
