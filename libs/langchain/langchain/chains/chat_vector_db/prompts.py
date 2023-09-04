# flake8: noqa
from langchain.prompts.prompt import PromptTemplate

_template = """Учитывая следующий разговор и последующий вопрос, переформулируй последующий вопрос так, чтобы он был самостоятельным вопросом.

История чата:
{chat_history}
Последующий вопрос: {question}
Самостоятельный вопрос:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

prompt_template = """Используй следующие части контекста, чтобы ответить на вопрос в конце. Если ты не знаешь ответа, просто скажи, что не знаешь, не пытайся придумать ответ.

{context}

Question: {question}
Полезный ответ:"""
QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
