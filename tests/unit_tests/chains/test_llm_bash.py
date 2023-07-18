"""Test LLM Bash functionality."""
import sys

import pytest

from langchain.chains.llm_bash.base import LLMBashChain
from langchain.chains.llm_bash.prompt import _PROMPT_TEMPLATE, BashOutputParser
from langchain.schema import OutputParserException
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
def output_parser() -> BashOutputParser:
    """Output parser for testing."""
    return BashOutputParser()


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="Test not supported on Windows"
)
def test_simple_question() -> None:
    """Test simple question that should not need python."""
    question = "Please write a bash script that prints 'Hello World' to the console."
    prompt = _PROMPT_TEMPLATE.format(question=question)
    queries = {prompt: "```bash\nexpr 1 + 1\n```"}
    fake_llm = FakeLLM(queries=queries)
    fake_llm_bash_chain = LLMBashChain.from_llm(fake_llm, input_key="q", output_key="a")
    output = fake_llm_bash_chain.run(question)
    assert output == "2\n"


def test_get_code(output_parser: BashOutputParser) -> None:
    """Test the parser."""
    code_lines = output_parser.parse(_SAMPLE_CODE)
    code = [c for c in code_lines if c.strip()]
    assert code == code_lines
    assert code == ["echo hello"]

    code_lines = output_parser.parse(_SAMPLE_CODE + _SAMPLE_CODE_2_LINES)
    assert code_lines == ["echo hello", "echo hello", "echo world"]


def test_parsing_error() -> None:
    """Test that LLM Output without a bash block raises an exce"""
    question = "Please echo 'hello world' to the terminal."
    prompt = _PROMPT_TEMPLATE.format(question=question)
    queries = {
        prompt: """
```text
echo 'hello world'
```
"""
    }
    fake_llm = FakeLLM(queries=queries)
    fake_llm_bash_chain = LLMBashChain.from_llm(fake_llm, input_key="q", output_key="a")
    with pytest.raises(OutputParserException):
        fake_llm_bash_chain.run(question)


def test_get_code_lines_mixed_blocks(output_parser: BashOutputParser) -> None:
    text = """
Unrelated text
```bash
echo hello
ls && pwd && ls
```

```python
print("hello")
```

```bash
echo goodbye
```
"""
    code_lines = output_parser.parse(text)
    assert code_lines == ["echo hello", "ls && pwd && ls", "echo goodbye"]


def test_get_code_lines_simple_nested_ticks(output_parser: BashOutputParser) -> None:
    """Test that backticks w/o a newline are ignored."""
    text = """
Unrelated text
```bash
echo hello
echo "```bash is in this string```"
```
"""
    code_lines = output_parser.parse(text)
    assert code_lines == ["echo hello", 'echo "```bash is in this string```"']
