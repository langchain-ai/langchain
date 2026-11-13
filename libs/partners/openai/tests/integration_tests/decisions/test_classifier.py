"""Integration tests for `OpenAIClassifier`."""

from __future__ import annotations

import base64
import io

import pytest
from langchain_core.decisions import Choice, Noul, NoulCriteria, Score
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_openai.decisions import OpenAIClassifier

pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_core._api.LangChainBetaWarning"
)

MODEL = "gpt-6-luna"

QUESTIONS: dict[str, Noul | Choice | Score] = {
    "damaged": Noul(
        instructions="Did the item arrive damaged?",
        criteria=NoulCriteria(true="The item is broken, cracked, or dented."),
    ),
    "department": Choice(
        instructions="Which team should handle this request?",
        criteria={
            "billing": "Payment or refund questions.",
            "shipping": "Delivery problems or damaged items.",
            "other": None,
        },
    ),
    "frustration": Score(
        instructions="How frustrated does the customer appear?",
        criteria=["Calm.", "Concerned but civil.", "Very angry."],
    ),
}


def test_invoke_all_question_types() -> None:
    response = OpenAIClassifier(model=MODEL).invoke(
        {
            "state": "The blue mug I ordered arrived cracked. Fix this today!",
            "questions": QUESTIONS,
        }
    )

    assert list(response.answers) == list(QUESTIONS)
    assert response.nouls["damaged"].noul > 0.5
    assert response.choices["department"].choice == "shipping"
    assert set(response.choices["department"].probabilities) == {
        "billing",
        "shipping",
        "other",
    }
    assert response.scores["frustration"].legend[2] == "Very angry."
    assert response.usage is not None
    assert response.usage["input_tokens"]
    assert response.request_id


async def test_ainvoke_with_tool_conversation() -> None:
    response = await OpenAIClassifier(model=MODEL).ainvoke(
        {
            "state": [
                HumanMessage("Where is my order?"),
                AIMessage(
                    "",
                    tool_calls=[
                        {"name": "track_order", "args": {"id": "A1"}, "id": "call_1"}
                    ],
                ),
                ToolMessage("Delivered yesterday.", tool_call_id="call_1"),
                AIMessage("It was delivered yesterday."),
                HumanMessage("It arrived smashed to pieces."),
            ],
            "questions": {"damaged": QUESTIONS["damaged"]},
        }
    )

    assert response.nouls["damaged"].noul > 0.5


def test_invoke_with_image() -> None:
    image = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    image.new("RGB", (64, 64), (255, 0, 0)).save(buffer, format="PNG")
    image_data = base64.b64encode(buffer.getvalue()).decode()

    response = OpenAIClassifier(model=MODEL).invoke(
        {
            "state": [
                HumanMessage(
                    content_blocks=[
                        {"type": "text", "text": "Product photo."},
                        {
                            "type": "image",
                            "base64": image_data,
                            "mime_type": "image/png",
                        },
                    ]
                )
            ],
            "questions": {
                "red": Noul(instructions="Is the photo a solid red image?"),
            },
        }
    )

    assert response.nouls["red"].noul > 0.5
