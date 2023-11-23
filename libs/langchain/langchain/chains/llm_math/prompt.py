# flake8: noqa
from langchain_core.prompts.prompt import PromptTemplate

_PROMPT_TEMPLATE = """Переведи математическую задачу в выражение, которое можно выполнить с помощью библиотеки numexpr в Python. Используй результат выполнения этого кода, чтобы ответить на вопрос.

Question: ${{Вопрос с математической задачей.}}
```text
${{однострочное математическое выражение, решающее задачу}}
```
...numexpr.evaluate(text)...
```output
${{Результат выполнения кода}}
```
Ответ: ${{Ответ}}

Начнем.

Question: Чему равно 37593 * 67?
```text
37593 * 67
```
...numexpr.evaluate("37593 * 67")...
```output
2518731
```
Ответ: 2518731

Question: 37593^(1/5)
```text
37593**(1/5)
```
...numexpr.evaluate("37593**(1/5)")...
```output
8.222831614237718
```
Ответ: 8.222831614237718

Question: {question}
"""

PROMPT = PromptTemplate(
    input_variables=["question"],
    template=_PROMPT_TEMPLATE,
)
