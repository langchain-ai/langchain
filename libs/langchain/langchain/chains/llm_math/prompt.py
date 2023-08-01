# flake8: noqa
from langchain.prompts.prompt import PromptTemplate

_PROMPT_TEMPLATE = """Translate a math problem into a expression that can be executed using the sympify function from the Python package Sympy.
Use the output of running this code to answer the question.
Sympify can handle most elementary mathematical expressions, including rounding, factorials, and calculus. It cannot handle string manipulation.

Question: ${{Question with math problem.}}
```text
${{single line mathematical expression that solves the problem}}
```
...sympify(text)...
```output
${{Output of running the code}}
```
Answer: ${{Answer}}

Begin.

Question: What is 37593 * 67?
```text
37593 * 67
```
...sympify("37593 * 67")...
```output
2518731
```
Answer: 2518731

Question: 37593^(1/5)
```text
37593**(1/5)
```
...sympify("37593**(1/5)")...
```output
8.222831614237718
```
Answer: 8.222831614237718

Question: {question}
"""

PROMPT = PromptTemplate(
    input_variables=["question"],
    template=_PROMPT_TEMPLATE,
)
