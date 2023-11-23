# flake8: noqa
from langchain_core.prompts.prompt import PromptTemplate

NAIVE_FIX = """Инструкции:
--------------
{instructions}
--------------
Завершение:
--------------
{completion}
--------------

Вышеуказанное Завершение не удовлетворяет ограничениям, указанным в Инструкциях.
Ошибка:
--------------
{error}
--------------

Пожалуйста, попробуй ещё раз. Отвечай только так, чтобы это удовлетворяло ограничениям, изложенным в Инструкциях:"""


NAIVE_FIX_PROMPT = PromptTemplate.from_template(NAIVE_FIX)
