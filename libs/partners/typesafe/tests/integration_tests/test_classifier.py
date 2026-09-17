"""Live integration tests for `TypeSafeClassifier`."""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_typesafe import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    TypeSafeClassifier,
)


def test_invoke_all_question_types() -> None:
    """Exercise the live sync API across Choice, Noul, and Score questions."""
    labels = {"billing", "technical", "sales"}
    classifier = TypeSafeClassifier(
        questions={
            "department": Choice(
                instructions="Which team should handle this request?",
                criteria={
                    "billing": "Payment or subscription issues.",
                    "technical": "Product bugs or integration failures.",
                    "sales": "Pricing or purchasing questions.",
                },
            ),
            "urgent": Noul(
                instructions="Does this message require an urgent response?"
            ),
            "frustration": Score(
                instructions="How frustrated does the customer appear?",
                criteria=[
                    "Calm and neutral.",
                    "Concerned but civil.",
                    "Very angry or using strong language.",
                ],
            ),
        }
    )

    try:
        response = classifier.invoke(
            {
                "message": (
                    "Stripe has failed to connect for three days. "
                    "Please help immediately."
                ),
                "account_tier": "enterprise",
            }
        )

        department = response.answers["department"]
        urgent = response.answers["urgent"]
        frustration = response.answers["frustration"]

        assert isinstance(department, ChoiceAnswer)
        assert department.choice in labels
        assert set(department.probabilities) == labels
        assert sum(department.probabilities.values()) == pytest.approx(1.0)

        assert isinstance(urgent, NoulAnswer)
        assert 0 <= urgent.noul <= 1

        assert isinstance(frustration, ScoreAnswer)
        assert 0 <= frustration.score <= 2
        assert set(frustration.legend) == {0, 1, 2}
        assert set(frustration.probabilities) == {0, 1, 2}
        assert sum(frustration.probabilities.values()) == pytest.approx(1.0)

        assert response.model.startswith("jev-")
        assert response.request_id
        assert isinstance(response.usage.input_tokens, int)
        assert isinstance(response.usage.output_tokens, int)
    finally:
        if classifier.client is not None:
            classifier.client.close()
        if classifier.async_client is not None:
            asyncio.run(classifier.async_client.aclose())


async def test_ainvoke_with_nested_messages() -> None:
    """Exercise the live async API with messages nested in structured state."""
    classifier = TypeSafeClassifier(
        questions={
            "needs_support": Noul(
                instructions="Does the user need help resolving a technical problem?"
            )
        }
    )

    try:
        response = await classifier.ainvoke(
            {
                "conversation": [
                    SystemMessage("You are reviewing a customer support conversation."),
                    HumanMessage(
                        "The integration crashes every time I connect Stripe. "
                        "Can someone help?"
                    ),
                ],
                "account": {
                    "tier": "enterprise",
                    "failed_attempts": 3,
                    "trial": False,
                    "notes": None,
                },
            }
        )

        needs_support = response.answers["needs_support"]
        assert isinstance(needs_support, NoulAnswer)
        assert 0 <= needs_support.noul <= 1
        assert response.model.startswith("jev-")
        assert response.request_id
    finally:
        if classifier.async_client is not None:
            await classifier.async_client.aclose()
        if classifier.client is not None:
            classifier.client.close()
