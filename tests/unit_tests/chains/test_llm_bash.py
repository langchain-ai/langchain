"""Test LLM Bash functionality."""
import sys

import pytest

from langchain.chains.llm_bash.base import LLMBashChain
from langchain.chains.llm_bash.prompt import _PROMPT_TEMPLATE
from tests.unit_tests.llms.fake_llm import FakeLLM


_SAMPLE_CODE = """
Unrelated text
```bash
echo hello
```
Unrelated text
"""


_SAMPLE_CODE_2_LINES = """
Unrelated text
```bash
echo hello

echo world
```
Unrelated text
"""


@pytest.fixture
def fake_llm_bash_chain() -> LLMBashChain:
    """Fake LLM Bash chain for testing."""
    question = "Please write a bash script that prints 'Hello World' to the console."
    prompt = _PROMPT_TEMPLATE.format(question=question)
    queries = {prompt: "```bash\nexpr 1 + 1\n```"}
    fake_llm = FakeLLM(queries=queries)
    return LLMBashChain(llm=fake_llm, input_key="q", output_key="a")


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="Test not supported on Windows"
)
def test_simple_question(fake_llm_bash_chain: LLMBashChain) -> None:
    """Test simple question that should not need python."""
    question = "Please write a bash script that prints 'Hello World' to the console."
    output = fake_llm_bash_chain.run(question)
    assert output == "2\n"


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="Test not supported on Windows"
)
def test_get_code_blocks(fake_llm_bash_chain) -> None:
    """Test the parser."""
    assert len(fake_llm_bash_chain.get_code_blocks(_SAMPLE_CODE)) == 1
    assert len(fake_llm_bash_chain.get_code_blocks(_SAMPLE_CODE + _SAMPLE_CODE_2_LINES)) == 2


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="Test not supported on Windows"
)
def test_get_code_blocks(fake_llm_bash_chain) -> None:
    """Test the parser."""
    assert len(fake_llm_bash_chain.get_code_blocks(_SAMPLE_CODE)) == 1
    assert len(fake_llm_bash_chain.get_code_blocks(_SAMPLE_CODE + _SAMPLE_CODE_2_LINES)) == 2


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="Test not supported on Windows"
)
def test_get_code(fake_llm_bash_chain) -> None:
    """Test the parser."""
    code_lines = fake_llm_bash_chain.get_code_lines(_SAMPLE_CODE)
    code = [c for c in code_lines if c]
    assert code == ["echo hello"]

    code_lines = fake_llm_bash_chain.get_code_lines(_SAMPLE_CODE + _SAMPLE_CODE_2_LINES)
    code = [c for c in code_lines if c]
    assert code == ["echo hello", "echo hello", "echo world"]
