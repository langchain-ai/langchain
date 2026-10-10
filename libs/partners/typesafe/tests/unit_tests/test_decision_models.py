"""Wire and conformance checks for the additive TypeSafe decision adapter."""

from __future__ import annotations

import json
from typing import Any

import httpx2
import pytest
from langchain_core.decision_models import (
    BaseDecisionModel,
    Choice,
    DecisionLevel,
    DecisionOption,
    DecisionRequest,
    DecisionResponseValidationError,
    Noul,
    NoulAnswer,
    Score,
)
from langchain_core.exceptions import ModelInvalidRequestError, ModelRateLimitError
from langchain_core.messages import HumanMessage
from langchain_tests.unit_tests.decision_models import DecisionModelUnitTests

from langchain_typesafe import TypeSafeClassifier
from langchain_typesafe._canonical import _TypeSafeDecisionModel


@pytest.fixture
def payload() -> dict[str, Any]:
    """Return a native mixed-question response fixture."""
    # Hand-authored wire fixture, not a captured live inference.
    return {
        "model": "resolved-test-model",
        "gateway_request_id": "gw-test",
        "cost": {"amount": 0.001, "currency": "USD"},
        "answers": {
            "urgent": {"type": "noul", "noul": 0.95},
            "team": {
                "type": "choice",
                "choice": "technical",
                "probabilities": {"technical": 0.9, "billing": 0.1},
                "confidence": 0.8,
            },
            "severity": {
                "type": "score",
                "score": 1.25,
                "legend": {"0": "calm", "1": "frustrated", "2": "angry"},
                "probabilities": {"0": 0.1, "1": 0.55, "2": 0.35},
                "confidence": 0.7,
            },
        },
        "usage": {"input_tokens": 42, "output_tokens": 0, "cached_tokens": 10},
    }


@pytest.fixture
async def native(payload: dict[str, Any]) -> Any:
    """Create a configured instance with offline sync and async transports."""

    def handler(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            payload.get("_status", 200),
            json=payload,
            headers={"x-typesafe-request-id": "req-test"},
        )

    with httpx2.Client(transport=httpx2.MockTransport(handler)) as client:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(handler)
        ) as async_client:
            yield TypeSafeClassifier(
                api_key="test-key",
                model="test-model",
                client=client,
                async_client=async_client,
            )


@pytest.fixture
def decision_request() -> DecisionRequest:
    """Return the same canonical request used by the OpenAI proof."""
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


class TestTypeSafeDecisionModel(DecisionModelUnitTests):
    """Apply mandatory shared checks to the TypeSafe adapter."""

    _native: TypeSafeClassifier

    @pytest.fixture(autouse=True)
    def configure(self, native: TypeSafeClassifier) -> None:
        """Inject the fixture's native instance for every shared check."""
        self._native = native

    @property
    def decision_model_class(self) -> type[BaseDecisionModel]:
        """Return the adapter implementation under test."""
        return _TypeSafeDecisionModel

    @property
    def decision_model_params(self) -> dict[str, Any]:
        """Reuse the configured native instance and its clients."""
        return {"native": self._native}


def test_additive_adapter(
    native: TypeSafeClassifier, decision_request: DecisionRequest
) -> None:
    """Retain client identity, answer semantics, and reported metadata."""
    adapter = native.as_decision_model()
    assert isinstance(adapter, _TypeSafeDecisionModel)
    assert adapter.native is native
    result = adapter.invoke(decision_request)
    assert result.answers["urgent"] == NoulAnswer(probability=0.95)
    assert result.model == "resolved-test-model"
    assert result.usage.total_tokens == 42
    assert result.response_metadata["gateway_request_id"] == "gw-test"
    assert result.response_metadata["provider_usage"]["cached_tokens"] == 10
    assert result.response_metadata["request_id"] == "req-test"
    assert result.response_metadata["cost"]["currency"] == "USD"
    assert adapter.to_json()["type"] == "not_implemented"


def test_body_and_header_request_ids(
    payload: dict[str, Any],
    native: TypeSafeClassifier,
    decision_request: DecisionRequest,
) -> None:
    """Keep both request identifiers when body and transport IDs differ."""
    payload["request_id"] = "body-request-id"
    result = native.as_decision_model().invoke(decision_request)
    assert result.response_metadata["request_id"] == "body-request-id"
    assert (
        result.response_metadata["response_headers"]["x-typesafe-request-id"]
        == "req-test"
    )


async def test_exact_wire_body(
    payload: dict[str, Any], decision_request: DecisionRequest
) -> None:
    """Translate all three primitives identically in sync and async calls."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        expected: dict[str, Any] = {
            "model": "test-model",
            "state": decision_request["state"],
            "questions": {
                "urgent": {"type": "noul", "instructions": "Is this urgent?"},
                "team": {
                    "type": "choice",
                    "instructions": "Which team?",
                    "criteria": {"billing": None, "technical": None},
                },
                "severity": {
                    "type": "score",
                    "instructions": "How severe?",
                    "criteria": ["calm", "frustrated", "angry"],
                },
            },
        }
        assert json.loads(request.content) == expected
        return httpx2.Response(200, json=payload)

    with httpx2.Client(transport=httpx2.MockTransport(handler)) as client:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(handler)
        ) as async_client:
            native = TypeSafeClassifier(
                api_key="test-key",
                model="test-model",
                client=client,
                async_client=async_client,
            )
            adapter = native.as_decision_model()
            adapter.invoke(decision_request)
            await adapter.ainvoke(decision_request)


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "extra_id",
        "wrong_type",
        "missing_option",
        "legend",
        "mass",
        "bad_usage",
    ],
)
def test_invalid_wire_response(
    case: str,
    payload: dict[str, Any],
    native: TypeSafeClassifier,
    decision_request: DecisionRequest,
) -> None:
    """Reject malformed and semantically mismatched successful responses."""
    if case == "missing":
        del payload["answers"]["urgent"]
    elif case == "extra_id":
        payload["answers"]["extra"] = payload["answers"]["urgent"]
    elif case == "wrong_type":
        payload["answers"]["urgent"]["type"] = "predicate"
    elif case == "missing_option":
        del payload["answers"]["team"]["probabilities"]["billing"]
    elif case == "legend":
        payload["answers"]["severity"]["legend"]["0"] = "invented"
    elif case == "mass":
        payload["answers"]["team"]["probabilities"]["billing"] = 0.5
    else:
        payload["usage"]["input_tokens"] = True
    with pytest.raises(DecisionResponseValidationError):
        native.as_decision_model().invoke(decision_request)


async def test_duplicate_json_keys_rejected() -> None:
    """Reject duplicate keys before a JSON mapping can discard them."""

    def handler(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            content='{"answers":{"q":{"type":"noul","noul":0.1},"q":{"type":"noul","noul":0.9}}}',
        )

    with httpx2.Client(transport=httpx2.MockTransport(handler)) as client:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(handler)
        ) as async_client:
            native = TypeSafeClassifier(
                api_key="test-key", client=client, async_client=async_client
            )
            with pytest.raises(DecisionResponseValidationError, match="duplicate"):
                native.as_decision_model().invoke(
                    {"state": "hello", "questions": {"q": Noul(instructions="Hi?")}}
                )


def test_boolean_options_rejected(native: TypeSafeClassifier) -> None:
    """Reject boolean options instead of coercing them into string keys."""
    with pytest.raises(ModelInvalidRequestError, match="boolean"):
        native.as_decision_model().invoke(
            {
                "state": "hello",
                "questions": {
                    "q": Choice(
                        instructions="Which?",
                        options=[
                            DecisionOption(value=True),
                            DecisionOption(value="true"),
                        ],
                    )
                },
            }
        )


def test_media_rejected(native: TypeSafeClassifier) -> None:
    """Reject nested message media instead of stripping evidence."""
    with pytest.raises(ModelInvalidRequestError, match="media"):
        native.as_decision_model().invoke(
            {
                "state": {
                    "nested": HumanMessage(
                        content=[
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,test"},
                            }
                        ]
                    )
                },
                "questions": {"q": Noul(instructions="Hi?")},
            }
        )


def test_unknown_usage_and_abstention(
    payload: dict[str, Any],
    native: TypeSafeClassifier,
    decision_request: DecisionRequest,
) -> None:
    """Preserve unreported counts and native abstention/confidence provenance."""
    del payload["usage"]
    payload["answers"]["team"]["abstained"] = True
    payload["answers"]["team"]["confidence_method"] = "entropy"
    result = native.as_decision_model().invoke(decision_request)
    assert result.usage.total_tokens is None
    assert result.answers["team"].abstained is True
    assert result.answers["team"].response_metadata["confidence_method"] == "entropy"


def test_transport_error_preserved(
    payload: dict[str, Any],
    native: TypeSafeClassifier,
    decision_request: DecisionRequest,
) -> None:
    """Keep the integration's shared model error categories."""
    payload["_status"] = 429
    with pytest.raises(ModelRateLimitError):
        native.as_decision_model().invoke(decision_request)
