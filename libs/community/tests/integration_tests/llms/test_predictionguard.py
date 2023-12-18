"""Test Prediction Guard API wrapper."""

from langchain_community.llms.predictionguard import PredictionGuard


def test_predictionguard_call() -> None:
    """Test valid call to prediction guard."""
    llm = PredictionGuard(model="OpenAI-gpt-3.5-turbo-instruct")
    output = llm("Say foo:")
    assert isinstance(output, str)
