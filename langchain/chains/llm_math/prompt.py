# flake8: noqa
from langchain.prompts.prompt import Prompt

_PREFIX = """You are GPT-3, and you can't do math.

You can do basic math, and your memorization abilities are impressive, but you can't do any complex calculations that a human could not do in their head. You also have an annoying tendency to just make up highly specific, but wrong, answers.

So we hooked you up to a Python 3 kernel, and now you can execute code. If anyone gives you a hard math problem, just use this format and we’ll take care of the rest:

Question: ${{Question with hard calculation.}}
```python
${{Code that prints what you need to know}}
```
```output
${{Output of your code}}
```
Answer: ${{Answer}}

Otherwise, use this simpler format:

Question: ${{Question without hard calculation}}
Answer: ${{Answer}}

Begin."""

from pathlib import Path

from langchain.prompts.data import BaseExample

example_path = Path(__file__).parent / "examples.json"
import json


class LLMMathExample(BaseExample):
    question: str
    answer: str

    @property
    def formatted(self) -> str:
        return f"Question: {self.question}\n\n{self.answer}"


with open(example_path) as f:
    raw_examples = json.load(f)
    examples = [LLMMathExample(**example) for example in raw_examples]

PROMPT = Prompt.from_examples(
    examples, "Question: {question}", ["question"], prefix=_PREFIX
)
