"""Test LLM Math functionality."""

import pytest

from langchain_experimental.llm_symbolic_math.base import (
    LLMSymbolicMathChain,
)
from langchain_experimental.llm_symbolic_math.prompt import (
    _PROMPT_TEMPLATE,
)
from tests.unit_tests.fake_llm import FakeLLM

try:
    import sympy
except ImportError:
    pytest.skip("sympy not installed", allow_module_level=True)


@pytest.fixture
def fake_llm_symbolic_math_chain() -> LLMSymbolicMathChain:
    """Fake LLM Math chain for testing."""
    queries = {
        _PROMPT_TEMPLATE.format(question="What is 1 plus 1?"): "Answer: 2",
        _PROMPT_TEMPLATE.format(
            question="What is the square root of 2?"
        ): "```text\nsqrt(2)\n```",
        _PROMPT_TEMPLATE.format(
            question="What is the limit of sin(x) / x as x goes to 0?"
        ): "```text\nlimit(sin(x)/x,x,0)\n```",
        _PROMPT_TEMPLATE.format(
            question="What is the integral of e^-x from 0 to infinity?"
        ): "```text\nintegrate(exp(-x), (x, 0, oo))\n```",
        _PROMPT_TEMPLATE.format(
            question="What are the solutions to this equation x**2 - x?"
        ): "```text\nsolveset(x**2 - x, x)\n```",
        _PROMPT_TEMPLATE.format(question="foo"): "foo",
    }
    fake_llm = FakeLLM(queries=queries)
    return LLMSymbolicMathChain.from_llm(fake_llm, input_key="q", output_key="a")


def test_simple_question(fake_llm_symbolic_math_chain: LLMSymbolicMathChain) -> None:
    """Test simple question that should not need python."""
    question = "What is 1 plus 1?"
    output = fake_llm_symbolic_math_chain.run(question)
    assert output == "Answer: 2"


def test_root_question(fake_llm_symbolic_math_chain: LLMSymbolicMathChain) -> None:
    """Test irrational number that should need sympy."""
    question = "What is the square root of 2?"
    output = fake_llm_symbolic_math_chain.run(question)
    assert output == f"Answer: {sympy.sqrt(2)}"


def test_limit_question(fake_llm_symbolic_math_chain: LLMSymbolicMathChain) -> None:
    """Test question about limits that needs sympy"""
    question = "What is the limit of sin(x) / x as x goes to 0?"
    output = fake_llm_symbolic_math_chain.run(question)
    assert output == "Answer: 1"


def test_integration_question(
    fake_llm_symbolic_math_chain: LLMSymbolicMathChain,
) -> None:
    """Test question about integration that needs sympy"""
    question = "What is the integral of e^-x from 0 to infinity?"
    output = fake_llm_symbolic_math_chain.run(question)
    assert output == "Answer: 1"


def test_solver_question(fake_llm_symbolic_math_chain: LLMSymbolicMathChain) -> None:
    """Test question about solving algebraic equations that needs sympy"""
    question = "What are the solutions to this equation x**2 - x?"
    output = fake_llm_symbolic_math_chain.run(question)
    assert output == "Answer: {0, 1}"


def test_error(fake_llm_symbolic_math_chain: LLMSymbolicMathChain) -> None:
    """Test question that raises error."""
    with pytest.raises(ValueError):
        fake_llm_symbolic_math_chain.run("foo")
