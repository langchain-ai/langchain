"""Test Prediction Guard API wrapper"""

import pytest

from langchain_community.chat_models.predictionguard import ChatPredictionGuard


def test_predictionguard_call() -> None:
    """Test a valid call to Prediction Guard."""
    chat = ChatPredictionGuard(
        model="Hermes-2-Pro-Llama-3-8B", max_tokens=100, temperature=1.0
    )

    messages = [
        (
            "system",
            "You are a helpful chatbot",
        ),
        ("human", "Tell me a joke."),
    ]

    output = chat.invoke(messages)
    assert isinstance(output.content, str)


def test_predictionguard_pii() -> None:
    chat = ChatPredictionGuard(
        model="Hermes-2-Pro-Llama-3-8B",
        predictionguard_input={
            "pii": "block",
        },
        max_tokens=100,
        temperature=1.0,
    )

    messages = [
        "Hello, my name is John Doe and my SSN is 111-22-3333",
    ]

    with pytest.raises(ValueError, match=r"Could not make prediction. pii detected"):
        chat.invoke(messages)


def test_predictionguard_stream() -> None:
    """Test a valid call with streaming to Prediction Guard"""

    chat = ChatPredictionGuard(
        model="Hermes-2-Pro-Llama-3-8B",
    )

    messages = [("system", "You are a helpful chatbot."), ("human", "Tell me a joke.")]

    num_chunks = 0
    for chunk in chat.stream(messages):
        assert isinstance(chunk.content, str)
        num_chunks += 1

    assert num_chunks > 0
