"""Unit tests for `OpenAIClassifier`."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import openai
import pytest
from langchain_core.decisions import (
    Box,
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    NoulCriteria,
    Question,
    Score,
    ScoreAnswer,
)
from langchain_core.exceptions import ModelRefusalError
from langchain_core.load import dumps, loads
from langchain_core.messages import (
    AIMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import JsonValue, SecretStr

from langchain_openai._compat import httpx
from langchain_openai.chat_models.base import OpenAIInvalidRequestError
from langchain_openai.decisions import OpenAIClassifier
from langchain_openai.decisions._classifier_input import (
    _to_classifier_input,
    _to_classifier_questions,
)

API_KEY = "test-api-key"
REQUEST_ID = "req_test"
MODEL = "gpt-6-luna"
IMAGE_URL = "data:image/png;base64,iVBORw0KGgo="

pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_core._api.LangChainBetaWarning"
)


def _questions() -> dict[str, Question]:
    return {
        "department": Choice(
            instructions="Which team should handle this request?",
            criteria={"billing": "Payment issues.", "technical": None},
        ),
        "urgent": Noul(instructions="Is this request urgent?"),
        "severity": Score(
            instructions="How severe is the issue?",
            criteria=["Cosmetic.", "Has a workaround.", {"level": "blocked"}],
        ),
    }


def _response_payload() -> dict[str, Any]:
    return {
        "model": MODEL,
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
                    {"value": 0, "label": "Cosmetic.", "probability": 0.1},
                    {"value": 1, "label": "Has a workaround.", "probability": 0.7},
                    {"value": 2, "label": '{"level": "blocked"}', "probability": 0.2},
                ],
                "confidence": 0.55,
            },
        ],
        "usage": {
            "input_tokens": 42,
            "input_tokens_details": {"cached_tokens": 10, "cache_write_tokens": 5},
            "output_tokens": 3,
            "output_tokens_details": {"reasoning_tokens": 2},
            "total_tokens": 45,
        },
    }


def _respond(
    payload: dict[str, Any], requests: list[dict[str, Any]] | None = None
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(json.loads(request.content))
        return httpx.Response(200, json=payload, headers={"x-request-id": REQUEST_ID})

    return handler


def _classifier(handler: Any) -> OpenAIClassifier:
    return OpenAIClassifier(
        model=MODEL,
        api_key=SecretStr(API_KEY),
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_missing_api_key_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_GATEWAY", raising=False)
    with pytest.raises(ValueError, match="API key is required"):
        OpenAIClassifier(model=MODEL)


def test_serialization_round_trip() -> None:
    classifier = OpenAIClassifier(model=MODEL, api_key=SecretStr(API_KEY))
    serialized = dumps(classifier)
    assert API_KEY not in serialized
    loaded = loads(
        serialized,
        secrets_map={"OPENAI_API_KEY": API_KEY},
        valid_namespaces=["langchain_openai"],
        allowed_objects=[OpenAIClassifier],
    )
    assert isinstance(loaded, OpenAIClassifier)
    assert loaded.model == MODEL


def test_invoke_sends_request_and_parses_response() -> None:
    requests: list[dict[str, Any]] = []
    classifier = _classifier(_respond(_response_payload(), requests))

    result = classifier.invoke({"state": "Stripe is down.", "questions": _questions()})

    assert requests == [
        {
            "model": MODEL,
            "input": "Stripe is down.",
            "questions": [
                {
                    "type": "choice",
                    "name": "department",
                    "instructions": "Which team should handle this request?",
                    "choices": [
                        {"value": "billing", "description": "Payment issues."},
                        {"value": "technical"},
                    ],
                },
                {
                    "type": "predicate",
                    "name": "urgent",
                    "instructions": "Is this request urgent?",
                },
                {
                    "type": "score",
                    "name": "severity",
                    "instructions": "How severe is the issue?",
                    "levels": [
                        {"label": "Cosmetic."},
                        {"label": "Has a workaround."},
                        {"label": '{"level": "blocked"}'},
                    ],
                },
            ],
        }
    ]
    assert result.model == MODEL
    assert result.request_id == REQUEST_ID
    assert result.choices["department"] == ChoiceAnswer(
        type="choice",
        choice="billing",
        probabilities={"billing": 0.9, "technical": 0.1},
        confidence=0.8,
    )
    assert result.nouls["urgent"] == NoulAnswer(type="noul", noul=0.95)
    assert result.scores["severity"] == ScoreAnswer(
        type="score",
        score=1.1,
        legend={0: "Cosmetic.", 1: "Has a workaround.", 2: {"level": "blocked"}},
        probabilities={0: 0.1, 1: 0.7, 2: 0.2},
        confidence=0.55,
    )
    assert result.usage == {
        "input_tokens": 42,
        "output_tokens": 3,
        "total_tokens": 45,
        "input_token_details": {"cache_read": 10, "cache_creation": 5},
        "output_token_details": {"reasoning": 2},
    }


async def test_ainvoke_uses_async_client() -> None:
    requests: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=_response_payload())

    classifier = OpenAIClassifier(
        model=MODEL,
        api_key=SecretStr(API_KEY),
        max_retries=0,
        http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    result = await classifier.ainvoke({"state": "hi", "questions": _questions()})

    assert len(requests) == 1
    assert result.nouls["urgent"].noul == 0.95


def test_noul_criteria_are_appended_to_instructions() -> None:
    questions = _to_classifier_questions(
        {
            "both": Noul(
                instructions="Is this action risky?",
                criteria=NoulCriteria(
                    true={"examples": ["delete"]}, false="Read-only."
                ),
            ),
            "true_only": Noul(
                instructions={"rule": "risky"},
                criteria=NoulCriteria(true="Moves money."),
            ),
            "empty": Noul(instructions="Is this risky?", criteria=NoulCriteria()),
        }
    )

    assert questions == [
        {
            "type": "predicate",
            "name": "both",
            "instructions": (
                "Is this action risky?\n\n"
                'Answer true if: {"examples": ["delete"]}\n'
                "Answer false if: Read-only."
            ),
        },
        {
            "type": "predicate",
            "name": "true_only",
            "instructions": [
                {"type": "input_json", "data": {"rule": "risky"}},
                {"type": "input_text", "text": "Answer true if: Moves money."},
            ],
        },
        {"type": "predicate", "name": "empty", "instructions": "Is this risky?"},
    ]


def test_refusal_raises_model_refusal_error() -> None:
    payload = _response_payload()
    payload["answers"][1] = {"type": "refusal", "name": "urgent"}
    classifier = _classifier(_respond(payload))

    with pytest.raises(ModelRefusalError, match="urgent"):
        classifier.invoke({"state": "hi", "questions": _questions()})


def test_mismatched_answer_count_is_invalid() -> None:
    payload = _response_payload()
    payload["answers"].pop()
    classifier = _classifier(_respond(payload))

    with pytest.raises(openai.APIResponseValidationError):
        classifier.invoke({"state": "hi", "questions": _questions()})


def test_bad_request_is_translated() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad questions"}})

    with pytest.raises(OpenAIInvalidRequestError):
        _classifier(handler).invoke({"state": "hi", "questions": _questions()})


def test_run_carries_model_identity() -> None:
    classifier = OpenAIClassifier(model=MODEL, api_key=SecretStr(API_KEY))
    metadata = classifier._traced_config(None)["metadata"]
    assert metadata["ls_provider"] == "openai"
    assert metadata["ls_model_name"] == MODEL


def test_box_is_rejected() -> None:
    with pytest.raises(ValueError, match="Box"):
        _to_classifier_questions({"where": Box(instructions="Locate the logo.")})


def test_single_label_choice_is_rejected() -> None:
    question = Choice(instructions="Which team?", criteria={"billing": None})
    with pytest.raises(ValueError, match="at least two"):
        _to_classifier_questions({"team": question})


def test_json_instructions() -> None:
    questions = _to_classifier_questions(
        {
            "object": Noul(instructions={"rule": "urgent"}),
            "array": Noul(instructions=["urgent", "blocking"]),
        }
    )
    assert questions[0]["instructions"] == [
        {"type": "input_json", "data": {"rule": "urgent"}}
    ]
    assert questions[1]["instructions"] == '["urgent", "blocking"]'


def test_images_in_instructions_are_sent_as_input_image() -> None:
    image: dict[str, JsonValue] = {
        "type": "image",
        "base64": "iVBORw0KGgo=",
        "mime_type": "image/png",
    }
    hosted: dict[str, JsonValue] = {"type": "image", "url": "https://example.com/a.png"}
    questions = _to_classifier_questions(
        {
            "image": Noul(instructions=image),
            "nested": Noul(
                instructions={"rule": "Matches the reference.", "ref": hosted}
            ),
        }
    )

    assert questions[0]["instructions"] == [
        {"type": "input_image", "image_url": IMAGE_URL}
    ]
    assert questions[1]["instructions"] == [
        {"type": "input_text", "text": '{"rule": "Matches the reference.", "ref": "'},
        {"type": "input_image", "image_url": "https://example.com/a.png"},
        {"type": "input_text", "text": '"}'},
    ]


@pytest.mark.parametrize(
    "question",
    [
        Noul(
            instructions={"type": "audio", "base64": "AAAA", "mime_type": "audio/wav"}
        ),
        Choice(
            instructions="Which?",
            criteria={
                "a": {"type": "image", "url": "https://example.com/a.png"},
                "b": "Other.",
            },
        ),
    ],
)
def test_unsupported_media_in_questions_is_rejected(question: Question) -> None:
    with pytest.raises(ValueError, match=r"Only text and images|only text"):
        _to_classifier_questions({"q": question})


def test_conversation_is_sent_as_native_items() -> None:
    state = [
        SystemMessage("Classify the last action."),
        HumanMessage(
            content=[
                {"type": "text", "text": "Clean up."},
                {"type": "image", "url": "https://example.com/a.png"},
            ]
        ),
        AIMessage(
            "Deleting.",
            tool_calls=[{"name": "rm", "args": {"path": "build/"}, "id": "call_1"}],
        ),
        ToolMessage("done", tool_call_id="call_1"),
    ]

    assert _to_classifier_input(state) == [
        {"role": "system", "content": "Classify the last action."},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Clean up."},
                {"type": "input_image", "image_url": "https://example.com/a.png"},
            ],
        },
        {"role": "assistant", "content": "Deleting."},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "rm",
            "arguments": '{"path": "build/"}',
        },
        {"type": "function_call_output", "call_id": "call_1", "output": "done"},
    ]


def test_object_state_is_sent_as_input_json() -> None:
    state: dict[str, Any] = {
        "ticket": "Stripe is down.",
        "history": [HumanMessage("hello")],
    }

    assert _to_classifier_input(state) == [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_json",
                    "data": {
                        "ticket": "Stripe is down.",
                        "history": [{"role": "user", "content": "hello"}],
                    },
                }
            ],
        }
    ]


def test_media_in_object_state_is_dropped_with_warning() -> None:
    image = {"type": "image_url", "image_url": {"url": IMAGE_URL}}
    state: dict[str, Any] = {
        "ticket": "T-1",
        "photo": {"type": "image", "base64": "iVBORw0KGgo=", "mime_type": "image/png"},
        "history": [HumanMessage(content=[{"type": "text", "text": "hi"}, image])],
    }

    with pytest.warns(UserWarning, match="Dropped 2 media block"):
        result = _to_classifier_input(state)

    assert result == [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_json",
                    "data": {
                        "ticket": "T-1",
                        "history": [{"role": "user", "content": "hi"}],
                    },
                }
            ],
        }
    ]


@pytest.mark.parametrize(
    "state",
    [
        [
            SystemMessage(
                content=[{"type": "image_url", "image_url": {"url": IMAGE_URL}}]
            )
        ],
        [
            HumanMessage(
                content=[
                    {"type": "file", "base64": "JVBER", "mime_type": "application/pdf"}
                ]
            )
        ],
        [AIMessage("", tool_calls=[{"name": "rm", "args": {}, "id": None}])],
        [ChatMessage(content="hi", role="critic")],
    ],
)
def test_unsupported_messages_are_rejected(state: list[Any]) -> None:
    with pytest.raises(ValueError, match="OpenAI Decisions API"):
        _to_classifier_input(state)
