"""Unit tests for `OpenAIDecisions`."""

from __future__ import annotations

import json
from typing import Any

import openai
import pytest
from langchain_core._api import LangChainBetaWarning
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.exceptions import ModelAuthenticationError
from langchain_core.load import dumps, loads
from pydantic import SecretStr, ValidationError

from langchain_openai._compat import httpx
from langchain_openai.chat_models.base import (
    OpenAIAuthenticationError,
    OpenAIInvalidRequestError,
)
from langchain_openai.decisions import (
    Choice,
    ChoiceAnswer,
    DecisionRequest,
    Level,
    OpenAIDecisions,
    Predicate,
    PredicateAnswer,
    RefusalAnswer,
    Score,
    ScoreAnswer,
)
from langchain_openai.decisions import base as base_module

API_KEY = "test-api-key"
REQUEST_ID = "req_test"
MODEL = "gpt-6-luna"

pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_core._api.LangChainBetaWarning"
)


class _RunTreeStub:
    """Stand-in for the LangSmith run tree that `_record_usage` writes to."""

    def __init__(self) -> None:
        self.extra: dict[str, Any] = {}


class _RunRecorder(BaseCallbackHandler):
    """Record the run type and metadata the runnable starts its run with."""

    def __init__(self) -> None:
        self.metadata: dict[str, Any] = {}
        self.input: Any = None
        self.run_type: str | None = None

    def on_chain_start(self, *args: Any, **kwargs: Any) -> None:
        self.input = args[1]
        self.metadata = kwargs.get("metadata") or {}
        self.run_type = kwargs.get("run_type")


def _response_payload() -> dict[str, Any]:
    return {
        "model": "gpt-6-luna",
        "answers": [
            {
                "type": "choice",
                "name": "department",
                "choice": "billing",
                "probabilities": [
                    {"value": "billing", "probability": 0.9},
                    {"value": "technical", "probability": 0.1},
                ],
                "confidence": 0.8,
            },
            {"type": "predicate", "name": "urgent", "probability": 0.95},
            {
                "type": "score",
                "name": "severity",
                "score": 1.1,
                "probabilities": [
                    {"value": 0, "label": "Cosmetic", "probability": 0.1},
                    {"value": 1, "label": "Workaround", "probability": 0.7},
                    {"value": 2, "label": "Blocked", "probability": 0.2},
                ],
                "confidence": 0.55,
            },
        ],
        "usage": {"input_tokens": 42, "output_tokens": 0, "total_tokens": 42},
    }


def _questions() -> dict[str, Choice | Predicate | Score]:
    return {
        "department": Choice(
            instructions="Which department?",
            choices={"billing": "Payments.", "technical": None},
        ),
        "urgent": Predicate(instructions="Is this urgent?"),
        "severity": Score(
            instructions="How severe?",
            levels=[
                "Cosmetic",
                Level(label="Workaround", description="Works."),
                "Blocked",
            ],
        ),
    }


def _request(input_: Any = "hello") -> DecisionRequest:
    return {"input": input_, "questions": _questions()}


def _ok(_: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200, json=_response_payload(), headers={"x-request-id": REQUEST_ID}
    )


def _decisions(handler: Any = _ok, **kwargs: Any) -> OpenAIDecisions:
    return OpenAIDecisions(
        model=MODEL,
        api_key=SecretStr(API_KEY),
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        **kwargs,
    )


def _async_decisions(handler: Any, **kwargs: Any) -> OpenAIDecisions:
    return OpenAIDecisions(
        model=MODEL,
        api_key=SecretStr(API_KEY),
        max_retries=0,
        http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        **kwargs,
    )


def test_is_beta() -> None:
    with pytest.warns(LangChainBetaWarning):
        OpenAIDecisions(model=MODEL, api_key=SecretStr(API_KEY))


def test_model_is_required() -> None:
    with pytest.raises(ValidationError, match="model"):
        OpenAIDecisions()  # type: ignore[call-arg]


def test_serialization_round_trip() -> None:
    decisions = OpenAIDecisions(
        model=MODEL, api_key=SecretStr(API_KEY), base_url="https://example.com/v1"
    )

    serialized = dumps(decisions)
    loaded = loads(
        serialized,
        secrets_map={"OPENAI_API_KEY": API_KEY},
        allowed_objects=[OpenAIDecisions],
    )

    assert API_KEY not in serialized
    assert isinstance(loaded, OpenAIDecisions)
    assert loaded.model == MODEL
    assert loaded.openai_api_base == "https://example.com/v1"
    assert dumps(loaded) == serialized


def test_load_does_not_read_api_key_from_environment() -> None:
    serialized = dumps(OpenAIDecisions(model=MODEL, api_key=SecretStr(API_KEY)))

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        loads(serialized, allowed_objects=[OpenAIDecisions])


def test_missing_api_key_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY")

    with pytest.raises(ValidationError, match="API key is required"):
        OpenAIDecisions(model=MODEL)


def test_api_key_from_environment() -> None:
    decisions = OpenAIDecisions(model=MODEL)

    assert decisions._client.api_key == "foo"


def test_score_requires_two_levels() -> None:
    with pytest.raises(ValidationError):
        Score(instructions="How severe?", levels=["only"])


def test_choice_requires_an_option() -> None:
    with pytest.raises(ValidationError):
        Choice(instructions="Which?", choices={})


def test_clients_are_created_lazily() -> None:
    decisions = _decisions()

    assert "_client" not in decisions.__dict__
    assert "_async_client" not in decisions.__dict__

    decisions.invoke(_request())

    assert "_client" in decisions.__dict__
    assert "_async_client" not in decisions.__dict__


def test_invoke_sends_request_and_parses_response() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers["authorization"]
        captured["body"] = json.loads(request.content)
        return _ok(request)

    result = _decisions(handler).invoke(_request())

    assert captured["url"] == "https://api.openai.com/v1/decisions"
    assert captured["auth"] == f"Bearer {API_KEY}"
    assert captured["body"] == {
        "model": "gpt-6-luna",
        "input": "hello",
        "questions": [
            {
                "type": "choice",
                "name": "department",
                "instructions": "Which department?",
                "choices": [
                    {"value": "billing", "description": "Payments."},
                    {"value": "technical"},
                ],
            },
            {"type": "predicate", "name": "urgent", "instructions": "Is this urgent?"},
            {
                "type": "score",
                "name": "severity",
                "instructions": "How severe?",
                "levels": [
                    {"label": "Cosmetic"},
                    {"label": "Workaround", "description": "Works."},
                    {"label": "Blocked"},
                ],
            },
        ],
    }
    assert result.choices["department"] == ChoiceAnswer(
        type="choice",
        choice="billing",
        probabilities={"billing": 0.9, "technical": 0.1},
        confidence=0.8,
    )
    assert result.predicates["urgent"] == PredicateAnswer(
        type="predicate", probability=0.95
    )
    assert result.scores["severity"] == ScoreAnswer(
        type="score",
        score=1.1,
        legend={0: "Cosmetic", 1: "Workaround", 2: "Blocked"},
        probabilities={0: 0.1, 1: 0.7, 2: 0.2},
        confidence=0.55,
    )
    assert result.usage.input_tokens == 42
    assert result.request_id == REQUEST_ID


def test_boolean_choice_values_round_trip() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "gpt-6-luna",
                "answers": [
                    {
                        "type": "choice",
                        "name": "flag",
                        "choice": True,
                        "probabilities": [
                            {"value": True, "probability": 0.98},
                            {"value": False, "probability": 0.02},
                        ],
                        "confidence": 0.96,
                    }
                ],
            },
        )

    result = _decisions(handler).invoke(
        {
            "input": "hi",
            "questions": {
                "flag": Choice(instructions="Flag?", choices={True: None, False: None})
            },
        }
    )

    assert captured["body"]["questions"][0]["choices"] == [
        {"value": True},
        {"value": False},
    ]
    assert result.choices["flag"].choice is True
    assert result.choices["flag"].probabilities == {True: 0.98, False: 0.02}


def test_refusals_are_returned() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "gpt-6-luna",
                "answers": [{"type": "refusal", "name": "urgent"}],
            },
        )

    result = _decisions(handler).invoke(_request())

    assert isinstance(result.refusals["urgent"], RefusalAnswer)
    assert result.predicates == {}


def test_unknown_answer_types_are_ignored() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        payload = _response_payload()
        payload["answers"].append({"type": "future", "name": "new"})
        return httpx.Response(200, json=payload)

    result = _decisions(handler).invoke(_request())

    assert set(result.answers) == {"department", "urgent", "severity"}


def test_invalid_response_does_not_expose_body() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"secret": "leaked", "answers": "bad"})

    with pytest.raises(openai.APIResponseValidationError) as exc_info:
        _decisions(handler).invoke(_request())

    assert "leaked" not in str(exc_info.value)
    assert exc_info.value.body is None


def test_authentication_error_is_translated() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    with pytest.raises(OpenAIAuthenticationError) as exc_info:
        _decisions(handler).invoke(_request())

    assert isinstance(exc_info.value, ModelAuthenticationError)


def test_bad_request_is_translated() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400, json={"error": {"message": "Question names must be unique."}}
        )

    with pytest.raises(OpenAIInvalidRequestError):
        _decisions(handler).invoke(_request())


async def test_ainvoke_uses_async_client() -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _ok(request)

    decisions = _async_decisions(handler)
    result = await decisions.ainvoke(_request())

    assert captured["body"]["input"] == "hello"
    assert result.predicates["urgent"].probability == 0.95
    assert "_client" not in decisions.__dict__


async def test_ainvoke_translates_api_error() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    with pytest.raises(OpenAIAuthenticationError):
        await _async_decisions(handler).ainvoke(_request())


def test_callbacks_receive_run() -> None:
    class RecordingHandler(BaseCallbackHandler):
        starts = 0
        ends = 0

        def on_chain_start(self, *_: Any, **__: Any) -> None:
            self.starts += 1

        def on_chain_end(self, *_: Any, **__: Any) -> None:
            self.ends += 1

    callback = RecordingHandler()
    _decisions().invoke(_request(), config={"callbacks": [callback]})

    assert callback.starts == 1
    assert callback.ends == 1


def test_usage_is_recorded_on_the_active_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Token usage is written where LangSmith totals it.

    LangSmith only sums tokens for `llm` runs whose metadata carries
    `usage_metadata`, so both the payload and the run type are pinned.
    """
    stub = _RunTreeStub()
    monkeypatch.setattr(base_module, "get_current_run_tree", lambda: stub)
    recorder = _RunRecorder()

    _decisions().invoke(_request(), config={"callbacks": [recorder]})

    assert recorder.run_type == "llm"
    assert recorder.input["input"] == "hello"
    assert set(recorder.input["questions"]) == set(_questions())
    assert stub.extra["metadata"]["usage_metadata"] == {
        "input_tokens": 42,
        "output_tokens": 0,
        "total_tokens": 42,
    }


async def test_async_usage_is_recorded_on_the_active_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _RunTreeStub()
    monkeypatch.setattr(base_module, "get_current_run_tree", lambda: stub)

    await _async_decisions(_async_ok).ainvoke(_request())

    assert stub.extra["metadata"]["usage_metadata"]["total_tokens"] == 42


async def _async_ok(request: httpx.Request) -> httpx.Response:
    return _ok(request)


def test_run_carries_model_identity_without_losing_caller_metadata() -> None:
    recorder = _RunRecorder()

    _decisions().invoke(
        _request(),
        config={"callbacks": [recorder], "metadata": {"tenant": "acme"}},
    )

    assert recorder.metadata["tenant"] == "acme"
    assert recorder.metadata["ls_provider"] == "openai"
    assert recorder.metadata["ls_model_name"] == "gpt-6-luna"
    assert API_KEY not in json.dumps(recorder.metadata, default=str)


def test_untraced_invocation_is_unaffected() -> None:
    result = _decisions().invoke(_request())

    assert result.usage.input_tokens == 42
