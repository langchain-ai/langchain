"""Unit tests for `TypeSafeClassifier`."""

from __future__ import annotations

import json
from typing import Any

import httpx2
import pytest
import typesafe_sdk as ts
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.load import dumpd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import SecretStr, ValidationError

from langchain_typesafe import (
    Choice,
    Noul,
    RetryPolicy,
    Score,
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAPIResponseValidationError,
    TypeSafeAPITimeoutError,
    TypeSafeClassifier,
)

API_KEY = "test-api-key"
REQUEST_ID = "req_test"
NO_RETRIES = RetryPolicy(max_retries=0)


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


def _classifier(
    handler: Any,
    *,
    questions: dict[str, Any] | None = None,
    retry: RetryPolicy | None = None,
    **kwargs: Any,
) -> TypeSafeClassifier:
    """Build a classifier whose sync client answers from `handler`."""
    return TypeSafeClassifier(
        api_key=API_KEY,
        questions=questions if questions is not None else _questions(),
        client=ts.TypeSafeClient(
            api_key=API_KEY,
            transport=httpx2.MockTransport(handler),
            retry=retry,
            **kwargs,
        ),
    )


def _async_classifier(
    handler: Any,
    *,
    questions: dict[str, Any] | None = None,
    retry: RetryPolicy | None = None,
    **kwargs: Any,
) -> TypeSafeClassifier:
    """Build a classifier whose async client answers from `handler`."""
    return TypeSafeClassifier(
        api_key=API_KEY,
        questions=questions if questions is not None else _questions(),
        async_client=ts.AsyncTypeSafeClient(
            api_key=API_KEY,
            transport=httpx2.MockTransport(handler),
            retry=retry,
            **kwargs,
        ),
    )


def _urgent() -> dict[str, Noul]:
    return {"urgent": Noul(instructions="Is this urgent?")}


def test_questions_must_not_be_empty() -> None:
    """The classifier requires at least one question."""
    with pytest.raises(ValidationError):
        TypeSafeClassifier(api_key=API_KEY, questions={})


def test_question_dictionaries_are_converted() -> None:
    """Question dictionaries become the corresponding SDK question types."""
    # Questions loaded from JSON or YAML arrive as plain dictionaries.
    questions: dict[str, Any] = {
        "urgent": {"type": "noul", "instructions": "Is this urgent?"},
        "team": {
            "type": "choice",
            "instructions": "Who handles this?",
            "criteria": {"billing": None},
        },
        "severity": {
            "type": "score",
            "instructions": "How severe?",
            "criteria": ["low", "high"],
        },
    }
    classifier = TypeSafeClassifier(api_key=API_KEY, questions=questions)

    assert isinstance(classifier.questions["urgent"], ts.Noul)
    assert isinstance(classifier.questions["team"], ts.Choice)
    assert isinstance(classifier.questions["severity"], ts.Score)


def test_unknown_question_type_is_rejected() -> None:
    """A question dictionary with an unrecognized type fails validation."""
    questions: dict[str, Any] = {
        "urgent": {"type": "boolean", "instructions": "Urgent?"}
    }
    with pytest.raises(ValidationError, match="unsupported type"):
        TypeSafeClassifier(api_key=API_KEY, questions=questions)


def test_non_question_values_are_rejected() -> None:
    """Values that are neither questions nor question dictionaries are rejected."""
    questions: dict[str, Any] = {"urgent": "Is this urgent?"}
    with pytest.raises(ValidationError):
        TypeSafeClassifier(api_key=API_KEY, questions=questions)


@pytest.mark.parametrize("model", ["", "   "])
def test_model_must_not_be_empty(model: str) -> None:
    """The classifier rejects empty and whitespace-only model identifiers."""
    with pytest.raises(ValidationError):
        TypeSafeClassifier(api_key=API_KEY, model=model, questions=_urgent())


def test_invoke_sends_request_and_parses_response() -> None:
    """The sync runnable sends the expected wire payload and parses each answer."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url == "https://api.typesafe.ai/v1/systemone"
        assert request.headers["authorization"] == f"Bearer {API_KEY}"
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

    with _classifier(handler) as classifier:
        result = classifier.invoke({"message": "Stripe fails to connect."})

    assert result.request_id == REQUEST_ID
    assert result.model == "jev-latest"
    assert result.usage.input_tokens == 42
    assert result.usage.output_tokens == 12
    assert result.choices["department"].choice == "technical"
    assert result.choices["department"].probabilities == {
        "billing": 0.1,
        "technical": 0.9,
    }
    assert result.choices["department"].confidence == 0.8
    assert result.nouls["urgent"].noul == 0.95
    assert result.scores["frustration"].score == 1.25
    assert result.scores["frustration"].legend == {
        0: "calm",
        1: "frustrated",
        2: "angry",
    }
    assert result.scores["frustration"].probabilities == {0: 0.1, 1: 0.55, 2: 0.35}


def test_requests_identify_the_integration() -> None:
    """Requests from classifier-created clients are attributable to this package."""
    observed = httpx2.Headers()

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal observed
        observed = request.headers
        return httpx2.Response(200, json=_response_payload())

    classifier = TypeSafeClassifier(api_key=API_KEY, questions=_questions())
    # Exercise the lazily created client while keeping the request on a mock transport.
    classifier.client = ts.TypeSafeClient(
        api_key=API_KEY,
        transport=httpx2.MockTransport(handler),
        headers=classifier._client_kwargs()["headers"],
    )
    classifier.invoke("hello")

    assert observed["x-langchain-integration"].startswith("langchain-typesafe/")
    # The SDK owns `User-Agent` and overwrites it, so it stays the SDK's.
    assert observed["user-agent"].startswith("typesafe-sdk/")


def test_single_message_is_serialized_as_role_content_state() -> None:
    """A LangChain message becomes a Jev-friendly role/content object."""
    observed_state: Any = None

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal observed_state
        observed_state = json.loads(request.content)["state"]
        return httpx2.Response(200, json=_response_payload())

    with _classifier(handler, questions=_urgent()) as classifier:
        classifier.invoke(HumanMessage("Please help immediately."))

    assert observed_state == {
        "role": "user",
        "content": "Please help immediately.",
    }


def test_message_sequence_is_serialized_as_conversation_state() -> None:
    """A message sequence preserves system, user, and assistant roles for Jev."""
    observed_state: Any = None

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal observed_state
        observed_state = json.loads(request.content)["state"]
        return httpx2.Response(200, json=_response_payload())

    with _classifier(handler, questions=_urgent()) as classifier:
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


async def test_ainvoke_uses_async_client() -> None:
    """The async runnable sends requests through the injected async client."""

    async def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.headers["authorization"] == f"Bearer {API_KEY}"
        return httpx2.Response(200, json=_response_payload())

    async with _async_classifier(handler) as classifier:
        result = await classifier.ainvoke("Please help ASAP.")

    assert result.choices["department"].choice == "technical"


def test_clients_are_created_lazily_and_reused() -> None:
    """No client exists until it is needed, and the same one is reused after."""
    classifier = TypeSafeClassifier(api_key=API_KEY, questions=_urgent())

    assert classifier.client is None
    assert classifier.async_client is None

    created = classifier._sync_client()

    assert classifier.client is created
    assert classifier._sync_client() is created
    # A sync-only caller never pays for an async client.
    assert classifier.async_client is None

    classifier.close()


def test_configuration_reaches_created_clients() -> None:
    """Classifier configuration is forwarded to clients it creates."""
    classifier = TypeSafeClassifier(
        api_key=API_KEY,
        questions=_urgent(),
        base_url="https://gateway.typesafe.example",
        model="jev-1.13",
        timeout=12.5,
        retry=NO_RETRIES,
    )
    kwargs = classifier._client_kwargs()

    assert kwargs["api_key"] == API_KEY
    assert kwargs["base_url"] == "https://gateway.typesafe.example"
    assert kwargs["model"] == "jev-1.13"
    assert kwargs["timeout"] == 12.5
    assert kwargs["retry"] is NO_RETRIES


def test_injected_clients_are_preserved_and_not_closed() -> None:
    """The classifier never replaces or closes a client supplied by the caller."""
    client = ts.TypeSafeClient(api_key=API_KEY)
    classifier = TypeSafeClassifier(
        api_key=API_KEY,
        questions=_urgent(),
        client=client,
    )

    assert classifier._sync_client() is client

    classifier.close()

    assert classifier.client is client
    client.close()


def test_close_releases_a_created_client() -> None:
    """Closing the classifier closes only the client it created."""
    classifier = TypeSafeClassifier(api_key=API_KEY, questions=_urgent())
    classifier._sync_client()

    classifier.close()

    assert classifier.client is None


async def test_aclose_releases_a_created_async_client() -> None:
    """Closing the classifier closes only the async client it created."""
    classifier = TypeSafeClassifier(api_key=API_KEY, questions=_urgent())
    classifier._get_async_client()

    await classifier.aclose()

    assert classifier.async_client is None


async def test_ainvoke_translates_api_error() -> None:
    """The async runnable raises the package exception for HTTP failures."""

    async def handler(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(429, headers={"x-typesafe-request-id": REQUEST_ID})

    async with _async_classifier(
        handler,
        questions=_urgent(),
        retry=NO_RETRIES,
    ) as classifier:
        with pytest.raises(TypeSafeAPIError) as exc_info:
            await classifier.ainvoke("hello")

    assert exc_info.value.status == 429
    assert exc_info.value.request_id == REQUEST_ID


async def test_ainvoke_translates_connection_error() -> None:
    """The async runnable raises the package exception for transport failures."""

    async def handler(request: httpx2.Request) -> httpx2.Response:
        msg = "transport detail"
        raise httpx2.ConnectError(msg, request=request)

    async with _async_classifier(
        handler,
        questions=_urgent(),
        retry=NO_RETRIES,
    ) as classifier:
        with pytest.raises(TypeSafeAPIConnectionError):
            await classifier.ainvoke("hello")


async def test_ainvoke_translates_timeout_error() -> None:
    """The async runnable classifies timeouts separately from connection failures."""

    async def handler(request: httpx2.Request) -> httpx2.Response:
        msg = "request timed out"
        raise httpx2.ReadTimeout(msg, request=request)

    async with _async_classifier(
        handler,
        questions=_urgent(),
        retry=NO_RETRIES,
        timeout=6.0,
    ) as classifier:
        with pytest.raises(TypeSafeAPITimeoutError) as exc_info:
            await classifier.ainvoke("hello")

    assert exc_info.value.timeout is not None


def test_api_key_is_held_as_a_secret() -> None:
    """The API key is stored as `SecretStr` and kept out of representations."""
    classifier = TypeSafeClassifier(api_key=API_KEY, questions=_urgent())

    assert isinstance(classifier.api_key, SecretStr)
    assert classifier.api_key.get_secret_value() == API_KEY
    assert API_KEY not in repr(classifier)
    assert API_KEY not in json.dumps(classifier.model_dump(mode="json"))


def test_api_key_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """A key in `TYPESAFE_API_KEY` satisfies construction and reaches the SDK."""
    monkeypatch.setenv("TYPESAFE_API_KEY", API_KEY)
    observed: dict[str, str] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        observed["authorization"] = request.headers["authorization"]
        return httpx2.Response(200, json=_response_payload())

    classifier = TypeSafeClassifier(questions=_urgent())
    classifier.client = ts.TypeSafeClient(transport=httpx2.MockTransport(handler))
    classifier.invoke("hello")

    assert classifier.api_key is None
    assert observed["authorization"] == f"Bearer {API_KEY}"


def test_base_url_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """`TYPESAFE_BASE_URL` changes where requests are sent."""
    monkeypatch.setenv("TYPESAFE_BASE_URL", "https://gateway.typesafe.example")
    observed: dict[str, str] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        observed["url"] = str(request.url)
        return httpx2.Response(200, json=_response_payload())

    classifier = TypeSafeClassifier(api_key=API_KEY, questions=_urgent())
    classifier.client = ts.TypeSafeClient(
        api_key=API_KEY,
        transport=httpx2.MockTransport(handler),
    )
    classifier.invoke("hello")

    assert observed["url"] == "https://gateway.typesafe.example/v1/systemone"


def test_explicit_base_url_overrides_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit API root takes precedence over environment configuration."""
    monkeypatch.setenv("TYPESAFE_BASE_URL", "https://environment.example")

    classifier = TypeSafeClassifier(
        api_key=API_KEY,
        base_url="https://explicit.example",
        questions=_urgent(),
    )

    assert classifier._client_kwargs()["base_url"] == "https://explicit.example"


def test_missing_api_key_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing a classifier without credentials fails immediately."""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(ValidationError, match="TypeSafe API key is required"):
        TypeSafeClassifier(questions=_urgent())


def test_injected_client_supplies_its_own_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fully injected client pair carries its own credentials."""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    classifier = TypeSafeClassifier(
        questions=_urgent(),
        client=ts.TypeSafeClient(api_key=API_KEY),
        async_client=ts.AsyncTypeSafeClient(api_key=API_KEY),
    )

    assert classifier.api_key is None
    classifier.client.close()  # type: ignore[union-attr]


def test_invalid_response_is_translated() -> None:
    """Malformed successful responses raise a stable package exception."""

    def handler(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={"model": "jev-latest", "answers": []})

    with (
        _classifier(handler, questions=_urgent(), retry=NO_RETRIES) as classifier,
        pytest.raises(TypeSafeAPIResponseValidationError) as exc_info,
    ):
        classifier.invoke("hello")

    assert exc_info.value.field_path


def test_retries_are_enabled_by_default() -> None:
    """A retryable status is retried without the caller configuring anything."""
    attempts = 0

    def handler(_: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx2.Response(529)
        return httpx2.Response(200, json=_response_payload())

    with _classifier(
        handler,
        questions=_urgent(),
        retry=RetryPolicy(backoff_initial=0.0, backoff_max=0.0),
    ) as classifier:
        result = classifier.invoke("hello")

    assert attempts == 2
    assert result.nouls["urgent"].noul == 0.95


def test_retries_can_be_disabled() -> None:
    """`RetryPolicy(max_retries=0)` surfaces the first failure to the caller."""
    attempts = 0

    def handler(_: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        attempts += 1
        return httpx2.Response(529)

    with (
        _classifier(handler, questions=_urgent(), retry=NO_RETRIES) as classifier,
        pytest.raises(TypeSafeAPIError),
    ):
        classifier.invoke("hello")

    assert attempts == 1


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

    callback = RecordingHandler()
    with _classifier(handler) as classifier:
        classifier.invoke("hello", config={"callbacks": [callback]})

    assert callback.starts == 1
    assert callback.ends == 1


def test_batch_reuses_one_client() -> None:
    """Concurrent batch invocations share a single lazily created client."""

    def handler(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_response_payload())

    with _classifier(handler, questions=_urgent()) as classifier:
        client = classifier.client
        results = classifier.batch(["one", "two", "three"])

    assert classifier.client is client
    assert len(results) == 3
    assert all(result.nouls["urgent"].noul == 0.95 for result in results)


def test_serialization_renders_questions_faithfully() -> None:
    """A serialized classifier carries its questions, not a placeholder."""
    classifier = TypeSafeClassifier(api_key=API_KEY, questions=_questions())

    serialized = dumpd(classifier)
    questions = serialized["kwargs"]["questions"]

    assert questions["urgent"] == {"type": "noul", "instructions": "Is this urgent?"}
    assert questions["frustration"]["criteria"] == ["calm", "frustrated", "angry"]
    assert "not_implemented" not in json.dumps(serialized)


def test_serialized_questions_round_trip() -> None:
    """Serialized questions can rebuild an equivalent classifier."""
    classifier = TypeSafeClassifier(api_key=API_KEY, questions=_questions())

    rebuilt = TypeSafeClassifier(
        api_key=API_KEY,
        questions=dumpd(classifier)["kwargs"]["questions"],
    )

    assert rebuilt.questions == classifier.questions


def test_serialization_keeps_the_api_key_out() -> None:
    """The API key is never written into a serialized classifier."""
    classifier = TypeSafeClassifier(api_key=API_KEY, questions=_urgent())

    assert API_KEY not in json.dumps(dumpd(classifier))
