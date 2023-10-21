import pytest
from langchain.chains.sentimental_analysis.base import SentimentAnalysisChain
from tests.unit_tests.llms.fake_llm import FakeLLM


def test_custom_chain_behavior() -> None:
    """Test the behavior of your custom chain."""
    # Define test cases and expected outputs here
    # Example:
    data = {"question":"i love this movie"}
    fake_llm = FakeLLM()
    chain = SentimentAnalysisChain.from_llm(fake_llm)
    expected_output = "positive"
    output = chain.run(data)
    assert output == expected_output
    
    