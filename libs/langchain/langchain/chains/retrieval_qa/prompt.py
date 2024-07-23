# flake8: noqa
from langchain_core.prompts import PromptTemplate

prompt_template = """Используй информацию между тэгами BEGIN_CONTEXT и END_CONTEXT, чтобы ответить на вопросы пользователя.Если ты не знаешь ответа и в данной тебе информации между тэгами BEGIN_CONTEXT и END_CONTEXT ее нет, просто скажи, что не знаешь, не пытайся придумать ответ.

BEGIN_CONTEXT
{context}
END_CONTEXT

Question: {question}
Полезный ответ:"""  # noqa
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
