"""Unit tests for `ModelRouterMiddleware`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.language_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_typesafe import ModelChoice, ModelRouterMiddleware
from tests.unit_tests.conftest import (
    RUNTIME,
    RecordingTransport,
    answers_response,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.unit_tests.conftest import Handler

FAST = FakeMessagesListChatModel(responses=[AIMessage("fast")])
POWERFUL = FakeMessagesListChatModel(responses=[AIMessage("powerful")])


def _choices() -> dict[str, ModelChoice]:
    return {
        "fast": ModelChoice(model=FAST, criteria="Simple, well-scoped tasks."),
        "powerful": ModelChoice(model=POWERFUL, criteria="Complex reasoning."),
    }


def _route_answer(route: str) -> dict[str, Any]:
    return answers_response(
        {
            "model_route": {
                "type": "choice",
                "choice": route,
                "probabilities": {"fast": 0.1, "powerful": 0.9},
                "confidence": 0.8,
            }
        }
    )


def _router(
    clients: Callable[[Handler], dict[str, Any]],
    transport: RecordingTransport,
    **kwargs: Any,
) -> ModelRouterMiddleware:
    return ModelRouterMiddleware(
        choices=_choices(),
        instructions="Choose the least costly model suited to the task.",
        default_route="powerful",
        **clients(transport),
        **kwargs,
    )


def test_empty_choices_are_rejected() -> None:
    """The router requires at least one model choice."""
    with pytest.raises(ValueError, match="At least one model choice"):
        ModelRouterMiddleware(choices={}, instructions="x", default_route="fast")


def test_default_route_must_be_a_choice() -> None:
    """The default route has to name one of the configured choices."""
    with pytest.raises(ValueError, match="not present in `choices`"):
        ModelRouterMiddleware(
            choices=_choices(),
            instructions="x",
            default_route="missing",
        )


def test_question_carries_every_choice_as_criteria(
    clients: Callable[[Handler], dict[str, Any]],
) -> None:
    """Each route becomes a labeled option in a single `Choice` question."""
    transport = RecordingTransport(_route_answer("fast"))
    router = _router(clients, transport)

    router.before_agent({"messages": [HumanMessage("Rename a variable.")]}, RUNTIME)

    question = transport.questions["model_route"]
    assert question["type"] == "choice"
    assert question["criteria"] == {
        "fast": "Simple, well-scoped tasks.",
        "powerful": "Complex reasoning.",
    }


def test_selected_route_is_stored_in_state(
    clients: Callable[[Handler], dict[str, Any]],
) -> None:
    """The classified route is persisted for the rest of the run."""
    transport = RecordingTransport(_route_answer("fast"))
    router = _router(clients, transport)

    assert router.before_agent(
        {"messages": [HumanMessage("Rename a variable.")]},
        RUNTIME,
    ) == {"model_route": "fast"}


def test_only_the_latest_human_message_is_classified(
    clients: Callable[[Handler], dict[str, Any]],
) -> None:
    """Routing looks at the newest human turn, not the whole transcript."""
    transport = RecordingTransport(_route_answer("fast"))
    router = _router(clients, transport)

    router.before_agent(
        {
            "messages": [
                SystemMessage("You are an assistant."),
                HumanMessage("First request."),
                AIMessage("Working on it."),
                HumanMessage("Second request."),
            ]
        },
        RUNTIME,
    )

    assert transport.state == {"role": "user", "content": "Second request."}


def test_no_human_message_uses_the_default_route(
    clients: Callable[[Handler], dict[str, Any]],
) -> None:
    """Without a human turn there is nothing to classify."""
    transport = RecordingTransport(_route_answer("fast"))
    router = _router(clients, transport)

    result = router.before_agent({"messages": [SystemMessage("Setup.")]}, RUNTIME)

    assert result == {"model_route": "powerful"}
    assert transport.requests == []


def test_unconfigured_route_falls_back(
    clients: Callable[[Handler], dict[str, Any]],
) -> None:
    """A route TypeSafe returns that is not configured is not trusted."""
    transport = RecordingTransport(_route_answer("experimental"))
    router = _router(clients, transport)

    assert router.before_agent({"messages": [HumanMessage("Hi.")]}, RUNTIME) == {
        "model_route": "powerful"
    }


def test_classification_failure_falls_back(
    clients: Callable[[Handler], dict[str, Any]],
) -> None:
    """Routing fails open so a TypeSafe outage cannot stop the agent."""
    transport = RecordingTransport(500)
    router = _router(clients, transport)

    assert router.before_agent({"messages": [HumanMessage("Hi.")]}, RUNTIME) == {
        "model_route": "powerful"
    }


def test_failure_logs_no_provider_message(
    clients: Callable[[Handler], dict[str, Any]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Failure logs carry error metadata, not the provider's response body."""
    transport = RecordingTransport(500)
    router = _router(clients, transport)

    with caplog.at_level("WARNING"):
        router.before_agent({"messages": [HumanMessage("Hi.")]}, RUNTIME)

    assert "TypeSafeInternalServerError" in caplog.text
    assert "default route" in caplog.text


async def test_async_classification_stores_route(
    clients: Callable[[Handler], dict[str, Any]],
) -> None:
    """The async hook behaves like the sync one."""
    transport = RecordingTransport(_route_answer("fast"))
    router = _router(clients, transport)

    result = await router.abefore_agent(
        {"messages": [HumanMessage("Rename a variable.")]},
        RUNTIME,
    )

    assert result == {"model_route": "fast"}


async def test_async_classification_failure_falls_back(
    clients: Callable[[Handler], dict[str, Any]],
) -> None:
    """The async hook also fails open."""
    transport = RecordingTransport(500)
    router = _router(clients, transport)

    result = await router.abefore_agent({"messages": [HumanMessage("Hi.")]}, RUNTIME)

    assert result == {"model_route": "powerful"}


class _Request:
    """Minimal `ModelRequest` stand-in recording the overridden model."""

    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state
        self.model: Any = None

    def override(self, **kwargs: Any) -> _Request:
        self.model = kwargs["model"]
        return self


def test_stored_route_selects_the_model(
    clients: Callable[[Handler], dict[str, Any]],
) -> None:
    """Every model call in the run uses the route chosen up front."""
    transport = RecordingTransport(_route_answer("fast"))
    router = _router(clients, transport)

    request = router._route_request(_Request({"model_route": "fast"}))  # type: ignore[arg-type]

    assert request.model is FAST  # type: ignore[attr-defined]


@pytest.mark.parametrize("route", ["unknown", None, 42])
def test_unusable_stored_route_selects_the_default(
    clients: Callable[[Handler], dict[str, Any]],
    route: Any,
) -> None:
    """A missing or malformed stored route still produces a usable model."""
    transport = RecordingTransport(_route_answer("fast"))
    router = _router(clients, transport)

    request = router._route_request(_Request({"model_route": route}))  # type: ignore[arg-type]

    assert request.model is POWERFUL  # type: ignore[attr-defined]


def test_model_call_is_delegated_to_the_handler(
    clients: Callable[[Handler], dict[str, Any]],
) -> None:
    """Routing rewrites the request and then defers to the next handler."""
    transport = RecordingTransport(_route_answer("fast"))
    router = _router(clients, transport)
    seen: list[Any] = []

    def handler(request: Any) -> Any:
        seen.append(request.model)
        return "response"

    router.wrap_model_call(_Request({"model_route": "fast"}), handler)  # type: ignore[arg-type]

    assert seen == [FAST]
