"""Unit tests for `TypeSafeClassifier`."""

from __future__ import annotations

import json
from typing import Any

import httpx2
import pytest
from langchain_core._api import LangChainBetaWarning
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import SecretStr, ValidationError

from langchain_typesafe import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    TypeSafeClassifier,
    __version__,
)
from langchain_typesafe import classifier as classifier_module
from langchain_typesafe.client import (
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAPIResponseValidationError,
    TypeSafeAPITimeoutError,
)


class _RunTreeStub:
    """Stand-in for the LangSmith run tree that `_record_usage` writes to."""

    def __init__(self) -> None:
        self.extra: dict[str, Any] = {}


class _RunRecorder(BaseCallbackHandler):
    """Record the run type and metadata the classifier starts its run with."""

    def __init__(self) -> None:
        self.metadata: dict[str, Any] = {}
        self.run_type: str | None = None

    def on_chain_start(self, *_: Any, **kwargs: Any) -> None:
        self.metadata = kwargs.get("metadata") or {}
        self.run_type = kwargs.get("run_type")


API_KEY = "test-api-key"
REQUEST_ID = "req_test"


def _response_payload() -> dict[str, Any]:
    return {
        "model": "jev-latest",
        "answers": {
            "department": {
                "type": "choice",
                "choice": "technical",
                "probabilities": {"billing": 0.1, "technical": 0.9},
                "confidence": 0.8,
            },
            "urgent": {"type": "noul", "noul": 0.95},
            "frustration": {
                "type": "score",
                "score": 1.25,
                "legend": {"0": "calm", "1": "frustrated", "2": "angry"},
                "probabilities": {"0": 0.1, "1": 0.55, "2": 0.35},
                "confidence": 0.7,
            },
        },
        "usage": {"input_tokens": 42, "output_tokens": 12},
    }


def _questions() -> dict[str, Choice | Noul | Score]:
    return {
        "department": Choice(
            instructions="Which team should handle this?",
            criteria={"billing": "Payment issues", "technical": None},
        ),
        "urgent": Noul(instructions="Is this urgent?"),
        "frustration": Score(
            instructions="How frustrated is the customer?",
            criteria=["calm", "frustrated", "angry"],
        ),
    }


def test_classifier_is_beta() -> None:
    """Constructing the classifier warns that its API is in beta."""
    with pytest.warns(
        LangChainBetaWarning,
        match=r"The class `TypeSafeClassifier` is in beta\.",
    ):
        TypeSafeClassifier(
            api_key=API_KEY,
            questions={"urgent": Noul(instructions="Is this urgent?")},
        )


def test_questions_require_instructions() -> None:
    """Every TypeSafe question requires an explicit instruction."""
    with pytest.raises(ValidationError, match="instructions"):
        Noul.model_validate({})
    with pytest.raises(ValidationError, match="instructions"):
        Choice.model_validate({"criteria": {"billing": None}})
    with pytest.raises(ValidationError, match="instructions"):
        Score.model_validate({"criteria": ["low", "high"]})


def test_score_requires_two_levels() -> None:
    """A Score rubric must define at least two ordered levels."""
    with pytest.raises(ValidationError, match="at least 2"):
        Score(instructions="How urgent is this?", criteria=["low"])


@pytest.mark.parametrize("model", ["", "   "])
def test_model_must_not_be_empty(model: str) -> None:
    """The classifier rejects empty and whitespace-only model identifiers."""
    with pytest.raises(ValidationError):
        TypeSafeClassifier(
            api_key=API_KEY,
            model=model,
            questions={"urgent": Noul(instructions="Is this urgent?")},
        )


def test_invoke_sends_request_and_parses_response() -> None:
    """The sync runnable sends the expected wire payload and parses each answer."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url == "https://api.typesafe.ai/v1/systemone"
        assert request.headers["authorization"] == f"Bearer {API_KEY}"
        assert request.headers["user-agent"] == f"langchain-typesafe/{__version__}"
        payload = json.loads(request.content)
        assert payload == {
            "state": {"message": "Stripe fails to connect."},
            "model": "jev-latest",
            "questions": {
                "department": {
                    "type": "choice",
                    "criteria": {
                        "billing": "Payment issues",
                        "technical": None,
                    },
                    "instructions": "Which team should handle this?",
                },
                "urgent": {
                    "type": "noul",
                    "instructions": "Is this urgent?",
                },
                "frustration": {
                    "type": "score",
                    "criteria": ["calm", "frustrated", "angry"],
                    "instructions": "How frustrated is the customer?",
                },
            },
        }
        return httpx2.Response(
            200,
            json=_response_payload(),
            headers={"x-typesafe-request-id": REQUEST_ID},
        )

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    classifier = TypeSafeClassifier(
        api_key=API_KEY,
        questions=_questions(),
        client=client,
    )

    result = classifier.invoke({"message": "Stripe fails to connect."})

    assert result.request_id == REQUEST_ID
    assert result.usage.input_tokens == 42
    assert result.choices["department"] == ChoiceAnswer(
        type="choice",
        choice="technical",
        probabilities={"billing": 0.1, "technical": 0.9},
        confidence=0.8,
    )
    assert result.nouls["urgent"] == NoulAnswer(type="noul", noul=0.95)
    assert result.scores["frustration"] == ScoreAnswer(
        type="score",
        score=1.25,
        legend={0: "calm", 1: "frustrated", 2: "angry"},
        probabilities={0: 0.1, 1: 0.55, 2: 0.35},
        confidence=0.7,
    )
    client.close()


def test_single_message_is_serialized_as_role_content_state() -> None:
    """A LangChain message becomes a Jev-friendly role/content object."""
    observed_state: Any = None

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal observed_state
        observed_state = json.loads(request.content)["state"]
        return httpx2.Response(200, json=_response_payload())

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    classifier = TypeSafeClassifier(
        api_key=API_KEY,
        questions={"urgent": Noul(instructions="Is this urgent?")},
        client=client,
    )

    classifier.invoke(HumanMessage("Please help immediately."))

    assert observed_state == {
        "role": "user",
        "content": "Please help immediately.",
    }
    client.close()


def test_message_sequence_is_serialized_as_conversation_state() -> None:
    """A message sequence preserves system, user, and assistant roles for Jev."""
    observed_state: Any = None

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal observed_state
        observed_state = json.loads(request.content)["state"]
        return httpx2.Response(200, json=_response_payload())

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    classifier = TypeSafeClassifier(
        api_key=API_KEY,
        questions={"urgent": Noul(instructions="Is the user asking for help?")},
        client=client,
    )

    classifier.invoke(
        [
            SystemMessage("You are a support assistant."),
            HumanMessage("My integration is broken."),
            AIMessage("I can help troubleshoot it."),
        ]
    )

    assert observed_state == [
        {"role": "system", "content": "You are a support assistant."},
        {"role": "user", "content": "My integration is broken."},
        {"role": "assistant", "content": "I can help troubleshoot it."},
    ]
    client.close()


@pytest.mark.asyncio
async def test_ainvoke_uses_async_client() -> None:
    """The async runnable sends requests through the injected async client."""

    async def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.headers["authorization"] == f"Bearer {API_KEY}"
        return httpx2.Response(200, json=_response_payload())

    async_client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    classifier = TypeSafeClassifier(
        api_key=API_KEY,
        questions=_questions(),
        async_client=async_client,
    )

    result = await classifier.ainvoke("Please help ASAP.")

    assert result.choices["department"].choice == "technical"
    await async_client.aclose()


@pytest.mark.asyncio
async def test_missing_clients_are_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Initialization creates both clients with the configured timeout when omitted."""
    sync_client = httpx2.Client()
    async_client = httpx2.AsyncClient()
    observed_timeouts: list[float] = []

    def sync_factory(*, timeout: float) -> httpx2.Client:
        observed_timeouts.append(timeout)
        return sync_client

    def async_factory(*, timeout: float) -> httpx2.AsyncClient:
        observed_timeouts.append(timeout)
        return async_client

    monkeypatch.setattr(httpx2, "Client", sync_factory)
    monkeypatch.setattr(httpx2, "AsyncClient", async_factory)

    classifier = TypeSafeClassifier(
        api_key=API_KEY,
        questions={"urgent": Noul(instructions="Is this urgent?")},
        timeout=12.5,
    )

    assert classifier.client is sync_client
    assert classifier.async_client is async_client
    assert observed_timeouts == [12.5, 12.5]
    sync_client.close()
    await async_client.aclose()


@pytest.mark.asyncio
async def test_injected_clients_are_preserved() -> None:
    """Initialization does not replace clients configured by the caller."""
    client = httpx2.Client()
    async_client = httpx2.AsyncClient()

    classifier = TypeSafeClassifier(
        api_key=API_KEY,
        questions={"urgent": Noul(instructions="Is this urgent?")},
        client=client,
        async_client=async_client,
    )

    assert classifier.client is client
    assert classifier.async_client is async_client
    client.close()
    await async_client.aclose()


@pytest.mark.asyncio
async def test_ainvoke_translates_api_error() -> None:
    """The async runnable translates unsuccessful API responses."""

    async def handler(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            429,
            headers={"x-typesafe-request-id": REQUEST_ID},
        )

    async_client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    classifier = TypeSafeClassifier(
        api_key=API_KEY,
        questions={"urgent": Noul(instructions="Is this urgent?")},
        async_client=async_client,
    )

    with pytest.raises(TypeSafeAPIError) as exc_info:
        await classifier.ainvoke("hello")

    assert exc_info.value.status_code == 429
    assert exc_info.value.request_id == REQUEST_ID
    await async_client.aclose()


@pytest.mark.asyncio
async def test_ainvoke_translates_connection_error() -> None:
    """The async runnable translates HTTP transport failures."""

    async def handler(request: httpx2.Request) -> httpx2.Response:
        message = "sensitive transport detail"
        raise httpx2.ConnectError(message, request=request)

    async_client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    classifier = TypeSafeClassifier(
        api_key=API_KEY,
        questions={"urgent": Noul(instructions="Is this urgent?")},
        async_client=async_client,
    )

    with pytest.raises(TypeSafeAPIConnectionError, match="Unable to connect"):
        await classifier.ainvoke("hello")

    await async_client.aclose()


@pytest.mark.asyncio
async def test_ainvoke_translates_timeout_error() -> None:
    """The async runnable classifies HTTP timeouts separately from connections."""

    async def handler(request: httpx2.Request) -> httpx2.Response:
        message = "request timed out"
        raise httpx2.ReadTimeout(message, request=request)

    async_client = httpx2.AsyncClient(
        timeout=6.0,
        transport=httpx2.MockTransport(handler),
    )
    classifier = TypeSafeClassifier(
        api_key=API_KEY,
        questions={"urgent": Noul(instructions="Is this urgent?")},
        async_client=async_client,
    )

    with pytest.raises(TypeSafeAPITimeoutError) as exc_info:
        await classifier.ainvoke("hello")

    assert exc_info.value.timeout == async_client.timeout
    await async_client.aclose()


def test_api_key_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The classifier reads its API key from `TYPESAFE_API_KEY`."""
    monkeypatch.setenv("TYPESAFE_API_KEY", API_KEY)
    classifier = TypeSafeClassifier(
        questions={"urgent": Noul(instructions="Is this urgent?")}
    )
    assert isinstance(classifier.api_key, SecretStr)
    assert classifier.api_key.get_secret_value() == API_KEY


def test_base_url_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The classifier reads its API root from `TYPESAFE_BASE_URL`."""
    monkeypatch.setenv("TYPESAFE_BASE_URL", "https://gateway.typesafe.example")

    classifier = TypeSafeClassifier(
        api_key=API_KEY,
        questions={"urgent": Noul(instructions="Is this urgent?")},
    )

    assert classifier.base_url == "https://gateway.typesafe.example"


def test_explicit_base_url_overrides_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit API root takes precedence over environment configuration."""
    monkeypatch.setenv("TYPESAFE_BASE_URL", "https://environment.example")

    classifier = TypeSafeClassifier(
        api_key=API_KEY,
        base_url="https://explicit.example",
        questions={"urgent": Noul(instructions="Is this urgent?")},
    )

    assert classifier.base_url == "https://explicit.example"


def test_missing_api_key_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing a classifier without credentials fails before creating clients."""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(ValidationError, match="TypeSafe API key is required"):
        TypeSafeClassifier(questions={"urgent": Noul(instructions="Is this urgent?")})


def test_api_error_does_not_expose_response_body() -> None:
    """HTTP errors expose status and request ID, but not server response bodies."""

    def handler(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            401,
            json={"error": "secret diagnostic"},
            headers={"x-typesafe-request-id": REQUEST_ID},
        )

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    classifier = TypeSafeClassifier(
        api_key=API_KEY,
        questions={"urgent": Noul(instructions="Is this urgent?")},
        client=client,
    )

    with pytest.raises(TypeSafeAPIError) as exc_info:
        classifier.invoke("hello")

    assert exc_info.value.status_code == 401
    assert exc_info.value.request_id == REQUEST_ID
    assert "secret diagnostic" not in str(exc_info.value)
    client.close()


def test_connection_error_is_translated() -> None:
    """HTTP transport failures use the package exception hierarchy."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        message = "sensitive transport detail"
        raise httpx2.ConnectError(message, request=request)

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    classifier = TypeSafeClassifier(
        api_key=API_KEY,
        questions={"urgent": Noul(instructions="Is this urgent?")},
        client=client,
    )

    with pytest.raises(TypeSafeAPIConnectionError, match="Unable to connect"):
        classifier.invoke("hello")

    client.close()


def test_timeout_error_is_translated() -> None:
    """HTTP timeout failures use provider and LangChain timeout hierarchies."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        message = "request timed out"
        raise httpx2.ReadTimeout(message, request=request)

    client = httpx2.Client(
        timeout=7.5,
        transport=httpx2.MockTransport(handler),
    )
    classifier = TypeSafeClassifier(
        api_key=API_KEY,
        questions={"urgent": Noul(instructions="Is this urgent?")},
        client=client,
    )

    with pytest.raises(TypeSafeAPITimeoutError) as exc_info:
        classifier.invoke("hello")

    assert exc_info.value.timeout == client.timeout
    client.close()


def test_invalid_response_is_translated() -> None:
    """Malformed successful responses raise a stable package exception."""

    def handler(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={"model": "jev-latest", "answers": []})

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    classifier = TypeSafeClassifier(
        api_key=API_KEY,
        questions={"urgent": Noul(instructions="Is this urgent?")},
        client=client,
    )

    with pytest.raises(TypeSafeAPIResponseValidationError, match="Invalid response"):
        classifier.invoke("hello")

    client.close()


def test_callbacks_receive_classifier_run() -> None:
    """Invocation participates in the standard LangChain callback lifecycle."""

    class RecordingHandler(BaseCallbackHandler):
        starts = 0
        ends = 0

        def on_chain_start(self, *_: Any, **__: Any) -> None:
            self.starts += 1

        def on_chain_end(self, *_: Any, **__: Any) -> None:
            self.ends += 1

    def handler(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_response_payload())

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    callback = RecordingHandler()
    classifier = TypeSafeClassifier(
        api_key=API_KEY,
        questions=_questions(),
        client=client,
    )

    classifier.invoke("hello", config={"callbacks": [callback]})

    assert callback.starts == 1
    assert callback.ends == 1
    client.close()


def test_usage_is_recorded_on_the_active_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Token usage is written where LangSmith totals it.

    LangSmith only sums tokens for `llm` runs whose metadata carries
    `usage_metadata`, so both the payload and the run type are pinned.
    """
    stub = _RunTreeStub()
    monkeypatch.setattr(classifier_module, "get_current_run_tree", lambda: stub)

    def handler(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_response_payload())

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    recorder = _RunRecorder()
    classifier = TypeSafeClassifier(
        api_key=API_KEY,
        questions=_questions(),
        client=client,
    )

    classifier.invoke("hello", config={"callbacks": [recorder]})
    client.close()

    assert recorder.run_type == "llm"
    assert stub.extra["metadata"]["usage_metadata"] == {
        "input_tokens": 42,
        "output_tokens": 12,
        "total_tokens": 54,
    }


async def test_async_usage_is_recorded_on_the_active_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`ainvoke` records the same usage as `invoke`."""
    stub = _RunTreeStub()
    monkeypatch.setattr(classifier_module, "get_current_run_tree", lambda: stub)

    async def handler(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_response_payload())

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    classifier = TypeSafeClassifier(
        api_key=API_KEY,
        questions=_questions(),
        async_client=client,
    )

    await classifier.ainvoke("hello")
    await client.aclose()

    assert stub.extra["metadata"]["usage_metadata"]["total_tokens"] == 54


def test_run_carries_model_identity_without_losing_caller_metadata() -> None:
    """Identity tags LangSmith prices by are added alongside caller metadata."""

    def handler(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_response_payload())

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    recorder = _RunRecorder()
    classifier = TypeSafeClassifier(
        api_key=API_KEY,
        questions=_questions(),
        client=client,
    )

    classifier.invoke(
        "hello",
        config={"callbacks": [recorder], "metadata": {"tenant": "acme"}},
    )
    client.close()

    assert recorder.metadata["tenant"] == "acme"
    assert recorder.metadata["ls_provider"] == "typesafe"
    assert recorder.metadata["ls_model_name"] == "jev-latest"
    assert API_KEY not in json.dumps(recorder.metadata, default=str)


def test_untraced_invocation_is_unaffected() -> None:
    """With no tracer active, recording usage is a no-op rather than an error."""

    def handler(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_response_payload())

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    classifier = TypeSafeClassifier(
        api_key=API_KEY,
        questions=_questions(),
        client=client,
    )

    result = classifier.invoke("hello")
    client.close()

    assert result.usage.input_tokens == 42
