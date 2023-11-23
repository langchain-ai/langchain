# flake8: noqa
from langchain_core.prompts.prompt import PromptTemplate

API_URL_PROMPT_TEMPLATE = """Тебе дана следующая документация API:
{api_docs}
Используя эту документацию, сформируй полный URL API для ответа на вопрос пользователя.
Тебе следует построить URL API так, чтобы получить ответ, который будет как можно короче, но при этом содержать необходимую информацию для ответа на вопрос. Обрати внимание на то, чтобы исключить все ненужные данные в вызове API.

Question:{question}
URL API:"""

API_URL_PROMPT = PromptTemplate(
    input_variables=[
        "api_docs",
        "question",
    ],
    template=API_URL_PROMPT_TEMPLATE,
)

API_RESPONSE_PROMPT_TEMPLATE = (
    API_URL_PROMPT_TEMPLATE
    + """ {api_url}

Вот ответ от API:

{api_response}

Суммируй этот ответ, чтобы ответить на исходный вопрос.

Сумма:"""
)

API_RESPONSE_PROMPT = PromptTemplate(
    input_variables=["api_docs", "question", "api_url", "api_response"],
    template=API_RESPONSE_PROMPT_TEMPLATE,
)
