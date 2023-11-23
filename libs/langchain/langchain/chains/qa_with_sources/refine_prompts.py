# flake8: noqa
from langchain_core.prompts import PromptTemplate

DEFAULT_REFINE_PROMPT_TMPL = (
    "Исходный вопрос звучит так: {question}\n"
    "Мы предоставили существующий ответ, включая источники: {existing_answer}\n"
    "У нас есть возможность уточнить существующий ответ"
    "(только если это необходимо) с некоторым дополнительным контекстом ниже.\n"
    "------------\n"
    "{context_str}\n"
    "------------\n"
    "Учитывая новый контекст, уточни исходный ответ, чтобы лучше "
    "ответить на вопрос. "
    "Если ты обновляешь его, пожалуйста, обнови и источники. "
    "Если контекст не полезен, верни исходный ответ."
)
DEFAULT_REFINE_PROMPT = PromptTemplate(
    input_variables=["question", "existing_answer", "context_str"],
    template=DEFAULT_REFINE_PROMPT_TMPL,
)


DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "Информация контекста ниже. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Учитывая информацию контекста и не имея предварительных знаний, "
    "ответь на вопрос: {question}\n"
)
DEFAULT_TEXT_QA_PROMPT = PromptTemplate(
    input_variables=["context_str", "question"], template=DEFAULT_TEXT_QA_PROMPT_TMPL
)

EXAMPLE_PROMPT = PromptTemplate(
    template="Содержание: {page_content}\nИсточник: {source}",
    input_variables=["page_content", "source"],
)
