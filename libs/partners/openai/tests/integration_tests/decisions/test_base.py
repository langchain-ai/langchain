"""Integration tests for `OpenAIDecisions`."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_openai.decisions import (
    Choice,
    OpenAIDecisions,
    Predicate,
    Question,
    Score,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_core._api.LangChainBetaWarning"
)

MODEL = "gpt-6-luna"

QUESTIONS: dict[str, Question] = {
    "department": Choice(
        instructions="Which department should handle this complaint?",
        choices={
            "billing": "Payments, invoices, and refunds.",
            "technical": "Problems using the product.",
            "other": None,
        },
    ),
    "urgent": Predicate(instructions="Does the customer need urgent help?"),
    "severity": Score(
        instructions="How severe is this issue?",
        levels=["Cosmetic", "Workaround available", "Fully blocked"],
    ),
}


def test_invoke_all_question_types() -> None:
    response = OpenAIDecisions(model=MODEL).invoke(
        {"input": "I was charged twice for my order.", "questions": QUESTIONS}
    )

    assert response.choices["department"].choice == "billing"
    assert 0 <= response.predicates["urgent"].probability <= 1
    assert 0 <= response.scores["severity"].score <= 2
    assert response.scores["severity"].legend[2] == "Fully blocked"
    assert response.usage.input_tokens
    assert response.request_id


async def test_ainvoke_with_conversation() -> None:
    response = await OpenAIDecisions(model=MODEL).ainvoke(
        {
            "input": [
                SystemMessage("You are reviewing a support conversation."),
                HumanMessage("Production is down and we are losing money."),
                AIMessage("I'm escalating this now."),
            ],
            "questions": {"urgent": QUESTIONS["urgent"]},
        }
    )

    assert response.predicates["urgent"].probability > 0.5
