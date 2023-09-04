# flake8: noqa

PREFIX = """
Ты работаешь со spark dataframe в Python. Имя dataframe - `df`.
Тебе следует использовать инструменты ниже, чтобы ответить на заданный тебе вопрос:"""

SUFFIX = """
Это результат `print(df.first())`:
{df}

Начни!
Question: {input}
{agent_scratchpad}"""
