# flake8: noqa
from langchain_core.prompts import PromptTemplate

prompt_template = """Используй следующие части контекста, чтобы ответить на вопрос в конце. Если ты не знаешь ответа, просто скажи, что не знаешь, не пытайся придумать ответ.

{context}

Question: {question}
Полезный ответ:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
