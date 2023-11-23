# flake8: noqa
from langchain_core.prompts.prompt import PromptTemplate

_template = """Посмотри на историю чата:
{chat_history}
Пользователь задал вопрос {question}
Перепиши вопрос пользователя, заменив в нем местоимения на значения из контекста. Если в вопросе ничего не нужно менять, то просто перепиши его."""  # noqa: E501
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

prompt_template = """Используй следующие части контекста, чтобы ответить на вопрос в конце. Если ты не знаешь ответа, просто скажи, что не знаешь, не пытайся придумать ответ.

{context}

Question: {question}
Полезный ответ:"""  # noqa: E501
QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
