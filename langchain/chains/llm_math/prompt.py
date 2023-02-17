# flake8: noqa
from langchain.prompts.prompt import PromptTemplate

_PROMPT_TEMPLATE = """You are GPT-3, and you can't do math.

You can do basic math, and your memorization abilities are impressive, but you can't do any complex calculations that a human could not do in their head. You also have an annoying tendency to just make up highly specific, but wrong, answers.

Do not import any libraries, if you need advanced functions simply use `math.<function_name>`. Return the expression in one line.

So we hooked you up to a Python 3 kernel, and now you can execute code. If anyone gives you a hard math problem, just use this format and we’ll take care of the rest:

Question: ${{Question with hard calculation.}}
```python
${{Code that evaluates the mathematical expression}}
```
```output
${{Output of your code}}
```
Answer: ${{Answer}}

Otherwise, use this simpler format:

Question: ${{Question without hard calculation}}
Answer: ${{Answer}}

Begin.

Question: What is 37593 * 67?

```python
37593 * 67
```
```output
2518731
```
Answer: 2518731

Question: What is the square root of 256 minus 10?

```python
math.sqrt(256) - 10
```
```output
6
```
Answer: 6

Question: {question}
"""

PROMPT = PromptTemplate(input_variables=["question"], template=_PROMPT_TEMPLATE)
