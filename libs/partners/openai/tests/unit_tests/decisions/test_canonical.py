from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.decision_models import (
    BaseDecisionModel,
    Choice,
    ChoiceAnswer,
    DecisionLevel,
    DecisionOption,
    DecisionRequest,
    DecisionResponseValidationError,
    Noul,
    NoulAnswer,
    Score,
)
from langchain_core.exceptions import ModelInvalidRequestError, ModelRateLimitError
from langchain_core.messages import AIMessage, HumanMessage
from langchain_tests.unit_tests.decision_models import DecisionModelUnitTests
from pydantic import SecretStr

from langchain_openai._compat import httpx
from langchain_openai.decisions import OpenAIDecisions
from langchain_openai.decisions._canonical import _OpenAIDecisionModel


@pytest.fixture
def payload() -> dict[str, Any]:
    # Hand-authored wire fixture, not a captured live inference.
    return {
        "model": "resolved-test-model",
        "gateway_request_id": "gw-test",
        "cost": {"amount": 0.001, "currency": "USD"},
        "answers": [
            {"name": "urgent", "type": "predicate", "probability": 0.95},
            {
                "name": "team",
                "type": "choice",
                "choice": "technical",
                "probabilities": [
                    {"value": "technical", "probability": 0.9},
                    {"value": "billing", "probability": 0.1},
                ],
                "confidence": 0.8,
            },
            {
                "name": "severity",
                "type": "score",
                "score": 1.25,
                "probabilities": [
                    {"value": 0, "label": "calm", "probability": 0.1},
                    {"value": 1, "label": "frustrated", "probability": 0.55},
                    {"value": 2, "label": "angry", "probability": 0.35},
                ],
                "confidence": 0.7,
            },
        ],
        "usage": {"input_tokens": 42, "output_tokens": 0, "cached_tokens": 10},
    }


@pytest.fixture
async def native(payload: dict[str, Any]) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        status = payload.get("_status", 200)
        return httpx.Response(
            status, json=payload, headers={"x-request-id": "req-test"}
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as async_client:
            yield OpenAIDecisions(
                model="test-model",
                api_key=SecretStr("test-key"),
                max_retries=0,
                http_client=client,
                http_async_client=async_client,
            )


@pytest.fixture
def decision_request() -> DecisionRequest:
    return {
        "state": "Payments have failed for three days. Please help!",
        "questions": {
            "urgent": Noul(instructions="Is this urgent?"),
            "team": Choice(
                instructions="Which team?",
                options=[
                    DecisionOption(value="billing"),
                    DecisionOption(value="technical"),
                ],
            ),
            "severity": Score(
                instructions="How severe?",
                levels=[
                    DecisionLevel(label="calm"),
                    DecisionLevel(label="frustrated"),
                    DecisionLevel(label="angry"),
                ],
            ),
        },
    }


class TestOpenAIDecisionModel(DecisionModelUnitTests):
    _native: OpenAIDecisions

    @pytest.fixture(autouse=True)
    def configure(self, native: OpenAIDecisions) -> None:
        self._native = native

    @property
    def decision_model_class(self) -> type[BaseDecisionModel]:
        return _OpenAIDecisionModel

    @property
    def decision_model_params(self) -> dict[str, Any]:
        return {"native": self._native}


class Recorder(BaseCallbackHandler):
    def __init__(self) -> None:
        self.starts = 0
        self.ends = 0
        self.errors = 0

    def on_chain_start(self, *args: Any, **kwargs: Any) -> None:
        self.starts += 1

    def on_chain_end(self, *args: Any, **kwargs: Any) -> None:
        self.ends += 1

    def on_chain_error(self, *args: Any, **kwargs: Any) -> None:
        self.errors += 1


def test_additive_adapter(
    native: OpenAIDecisions, decision_request: DecisionRequest
) -> None:
    recorder = Recorder()
    adapter = native.as_decision_model()
    assert isinstance(adapter, BaseDecisionModel)
    assert isinstance(adapter, _OpenAIDecisionModel)
    assert adapter.native is native
    result = adapter.invoke(decision_request, {"callbacks": [recorder]})
    assert result.answers["urgent"] == NoulAnswer(probability=0.95)
    assert result.model == "resolved-test-model"
    assert result.usage.total_tokens == 42
    assert result.response_metadata["gateway_request_id"] == "gw-test"
    assert result.response_metadata["provider_usage"]["cached_tokens"] == 10
    assert result.response_metadata["request_id"] == "req-test"
    assert result.response_metadata["cost"]["currency"] == "USD"
    assert recorder.starts == recorder.ends == 1
    assert recorder.errors == 0
    assert adapter.to_json()["type"] == "not_implemented"


def test_body_and_header_request_ids(
    payload: dict[str, Any],
    native: OpenAIDecisions,
    decision_request: DecisionRequest,
) -> None:
    payload["request_id"] = "body-request-id"
    result = native.as_decision_model().invoke(decision_request)
    assert result.response_metadata["request_id"] == "body-request-id"
    assert result.response_metadata["response_headers"]["x-request-id"] == "req-test"


def test_exact_wire_body(
    payload: dict[str, Any], decision_request: DecisionRequest
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        expected: dict[str, Any] = {
            "model": "test-model",
            "input": decision_request["state"],
            "questions": [
                {
                    "name": "urgent",
                    "type": "predicate",
                    "instructions": "Is this urgent?",
                },
                {
                    "name": "team",
                    "type": "choice",
                    "instructions": "Which team?",
                    "choices": [{"value": "billing"}, {"value": "technical"}],
                },
                {
                    "name": "severity",
                    "type": "score",
                    "instructions": "How severe?",
                    "levels": [
                        {"label": "calm"},
                        {"label": "frustrated"},
                        {"label": "angry"},
                    ],
                },
            ],
        }
        assert json.loads(request.content) == expected
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        native = OpenAIDecisions(
            model="test-model", api_key=SecretStr("test-key"), http_client=client
        )
        native.as_decision_model().invoke(decision_request)


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "duplicate_id",
        "wrong_type",
        "duplicate_option",
        "missing_option",
        "score_index",
        "bad_usage",
        "unknown",
    ],
)
def test_invalid_wire_response(
    case: str,
    payload: dict[str, Any],
    native: OpenAIDecisions,
    decision_request: DecisionRequest,
) -> None:
    if case == "missing":
        payload["answers"].pop()
    elif case == "duplicate_id":
        payload["answers"].append(payload["answers"][0])
    elif case == "wrong_type":
        payload["answers"][0]["type"] = "noul"
    elif case == "duplicate_option":
        payload["answers"][1]["probabilities"].append(
            payload["answers"][1]["probabilities"][0]
        )
    elif case == "missing_option":
        payload["answers"][1]["probabilities"].pop()
    elif case == "score_index":
        payload["answers"][2]["probabilities"][0]["value"] = True
    elif case == "bad_usage":
        payload["usage"]["input_tokens"] = True
    else:
        payload["answers"][0]["type"] = "unknown"
    recorder = Recorder()
    with pytest.raises(DecisionResponseValidationError):
        native.as_decision_model().invoke(decision_request, {"callbacks": [recorder]})
    assert recorder.starts == recorder.errors == 1
    assert recorder.ends == 0


def test_refusal_and_unknown_usage(
    payload: dict[str, Any], native: OpenAIDecisions, decision_request: DecisionRequest
) -> None:
    payload["answers"][0] = {"name": "urgent", "type": "refusal", "reason": "policy"}
    del payload["usage"]
    result = native.as_decision_model().invoke(decision_request)
    assert result.answers["urgent"].type == "refusal"
    assert result.answers["urgent"].response_metadata["reason"] == "policy"
    assert result.usage.input_tokens is result.usage.total_tokens is None


def test_boolean_choice_identity(
    payload: dict[str, Any], native: OpenAIDecisions
) -> None:
    payload["answers"] = [
        {
            "name": "q",
            "type": "choice",
            "choice": True,
            "probabilities": [
                {"value": "true", "probability": 0.3},
                {"value": True, "probability": 0.7},
            ],
        }
    ]
    result = native.as_decision_model().invoke(
        {
            "state": "hello",
            "questions": {
                "q": Choice(
                    instructions="Which?",
                    options=[DecisionOption(value=True), DecisionOption(value="true")],
                )
            },
        }
    )
    answer = result.answers["q"]
    assert isinstance(answer, ChoiceAnswer)
    assert answer.value is True
    assert answer.selected_probability == 0.7


@pytest.mark.parametrize(
    "state",
    [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.png"},
                }
            ]
        ),
        {"nested": AIMessage(content=[{"type": "audio", "data": "test"}])},
    ],
)
def test_unsupported_media(
    native: OpenAIDecisions, decision_request: DecisionRequest, state: Any
) -> None:
    decision_request["state"] = state
    with pytest.raises(ModelInvalidRequestError):
        native.as_decision_model().invoke(decision_request)


def test_transport_error_preserved(
    payload: dict[str, Any], native: OpenAIDecisions, decision_request: DecisionRequest
) -> None:
    payload["_status"] = 429
    with pytest.raises(ModelRateLimitError):
        native.as_decision_model().invoke(decision_request)
