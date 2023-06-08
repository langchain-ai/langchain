"""Test LLM PAL functionality."""
import sys
sys.path.append(r"C:\projects\langchain")

import pytest

from langchain.chains.pal.base import PALChain
from langchain.chains.pal.math_prompt import MATH_PROMPT
from langchain.schema import OutputParserException
from tests.unit_tests.llms.fake_llm import FakeLLM

_SAMPLE_CODE = """
def solution():
    \"\"\"Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\"\"\"
    money_initial = 23
    bagels = 5
    bagel_cost = 3
    money_spent = bagels * bagel_cost
    money_left = money_initial - money_spent
    result = money_left
    return result
"""


_SAMPLE_CODE_2_LINES = """
Unrelated text
```bash
echo hello

echo world
```
Unrelated text
"""

FULL_CODE_VALIDATIONS = {'solution_function': 'solution', 'allow_imports': False, 'allow_non_solution_root_scope_expressions': False, 'allow_non_math_operations': False}

def test_simple_question() -> None:
    """Test simple question."""
    question = "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?"
    prompt = MATH_PROMPT.format(question=question)
    queries = {prompt: _SAMPLE_CODE}
    fake_llm = FakeLLM(queries=queries)
    fake_pal_chain = PALChain.from_math_prompt(fake_llm)
    output = fake_pal_chain.run(question)
    assert output == "8"


def test_get_code() -> None:
    """Test the validator."""
    PALChain.validate_code(_SAMPLE_CODE, FULL_CODE_VALIDATIONS)