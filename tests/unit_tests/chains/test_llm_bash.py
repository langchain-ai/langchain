"""Test LLM Bash functionality."""

import pytest

from langchain.chains.llm_bash.base import LLMBashChain
from langchain.chains.llm_bash.prompt import _PROMPT_TEMPLATE
from tests.unit_tests.llms.fake_llm import FakeLLM


@pytest.fixture
def fake_llm_bash_chain() -> LLMBashChain:
    """Fake LLM Bash chain for testing."""
    question = "Please write a bash script that prints 'Hello World' to the console."
    prompt = _PROMPT_TEMPLATE.format(question=question)
    queries = {prompt: "```bash\nexpr 1 + 1\n```"}
    fake_llm = FakeLLM(queries=queries)
    return LLMBashChain(llm=fake_llm, input_key="q", output_key="a")


def test_simple_question(fake_llm_bash_chain: LLMBashChain) -> None:
    """Test simple question that should not need python."""
    question = "Please write a bash script that prints 'Hello World' to the console."
    output = fake_llm_bash_chain.run(question)
    assert output == "2\n"
