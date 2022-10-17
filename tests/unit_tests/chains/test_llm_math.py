"""Test LLM Math functionality."""

import pytest

from langchain.chains.llm_math.base import LLMMathChain
from langchain.chains.llm_math.prompt import _PROMPT_TEMPLATE
from tests.unit_tests.llms.fake_llm import FakeLLM


@pytest.fixture
def fake_llm_math_chain() -> LLMMathChain:
    """Fake LLM Math chain for testing."""
    complex_question = _PROMPT_TEMPLATE.format(question="What is the square root of 2?")
    queries = {
        _PROMPT_TEMPLATE.format(question="What is 1 plus 1?"): "Answer: 2",
        complex_question: "```python\nprint(2**.5)\n```",
        _PROMPT_TEMPLATE.format(question="foo"): "foo",
    }
    fake_llm = FakeLLM(queries=queries)
    return LLMMathChain(llm=fake_llm, input_key="q", output_key="a")


def test_simple_question(fake_llm_math_chain: LLMMathChain) -> None:
    """Test simple question that should not need python."""
    question = "What is 1 plus 1?"
    output = fake_llm_math_chain.run(question)
    assert output == "Answer: 2"


def test_complex_question(fake_llm_math_chain: LLMMathChain) -> None:
    """Test complex question that should need python."""
    question = "What is the square root of 2?"
    output = fake_llm_math_chain.run(question)
    assert output == f"Answer: {2**.5}\n"


def test_error(fake_llm_math_chain: LLMMathChain) -> None:
    """Test question that raises error."""
    with pytest.raises(ValueError):
        fake_llm_math_chain.run("foo")
