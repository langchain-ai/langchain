"""Test Prediction Guard API wrapper."""

import pytest

from langchain_community.llms.predictionguard import PredictionGuard


def test_predictionguard_invoke() -> None:
    """Test valid call to prediction guard."""
    llm = PredictionGuard(model="Hermes-3-Llama-3.1-8B")  # type: ignore[call-arg]
    output = llm.invoke("Tell a joke.")
    assert isinstance(output, str)


def test_predictionguard_pii() -> None:
    llm = PredictionGuard(
        model="Hermes-3-Llama-3.1-8B",
        predictionguard_input={"pii": "block"},
        max_tokens=100,
        temperature=1.0,
    )

    messages = [
        "Hello, my name is John Doe and my SSN is 111-22-3333",
    ]

    with pytest.raises(ValueError, match=r"Could not make prediction. pii detected"):
        llm.invoke(messages)
