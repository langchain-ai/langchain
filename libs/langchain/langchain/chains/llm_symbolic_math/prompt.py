# flake8: noqa
from langchain.prompts.prompt import PromptTemplate

_PROMPT_TEMPLATE = """Переведи математическую задачу в выражение, которое можно выполнить с помощью библиотеки SymPy в Python. Используй результат выполнения этого кода, чтобы ответить на вопрос.

Question: ${{Вопрос с математической задачей.}}
```text
${{однострочное выражение sympy, которое решает задачу}}
```
...sympy.sympify(text, evaluate=True)...
```output
${{Результат выполнения кода}}
```
Ответ: ${{Ответ}}

Начнем.

Question: Каков предел sin(x) / x при x, стремящемся к 0
```text
limit(sin(x)/x, x, 0)
```
...sympy.sympify("limit(sin(x)/x, x, 0)")...
```output
1
```
Ответ: 1

Question: Каков интеграл от e^-x от 0 до бесконечности
```text
integrate(exp(-x), (x, 0, oo))
```
...sympy.sympify("integrate(exp(-x), (x, 0, oo))")...
```output
1
```

Question: Какие решения у этого уравнения x**2 - x?
```text
solveset(x**2 - x, x)
```
...sympy.sympify("solveset(x**2 - x, x)")...
```output
[0, 1]
```
Question: {question}
"""

PROMPT = PromptTemplate(
    input_variables=["question"],
    template=_PROMPT_TEMPLATE,
)
