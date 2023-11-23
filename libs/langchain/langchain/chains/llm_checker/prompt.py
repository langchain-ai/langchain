# flake8: noqa
from langchain_core.prompts.prompt import PromptTemplate

_CREATE_DRAFT_ANSWER_TEMPLATE = """{question}\n\n"""
CREATE_DRAFT_ANSWER_PROMPT = PromptTemplate(
    input_variables=["question"], template=_CREATE_DRAFT_ANSWER_TEMPLATE
)

_LIST_ASSERTIONS_TEMPLATE = """Вот утверждение:
{statement}
Составь список предположений, которые ты сделал, формулируя вышеуказанное утверждение.\n\n"""
LIST_ASSERTIONS_PROMPT = PromptTemplate(
    input_variables=["statement"], template=_LIST_ASSERTIONS_TEMPLATE
)

_CHECK_ASSERTIONS_TEMPLATE = """Вот список утверждений:
{assertions}
Для каждого утверждения определи, верно оно или нет. Если оно неверно, объясни почему.\n\n"""
CHECK_ASSERTIONS_PROMPT = PromptTemplate(
    input_variables=["assertions"], template=_CHECK_ASSERTIONS_TEMPLATE
)

_REVISED_ANSWER_TEMPLATE = """{checked_assertions}

Question: Учитывая вышеуказанные утверждения и проверки, как бы ты ответил на вопрос '{question}'?

Ответ:"""
REVISED_ANSWER_PROMPT = PromptTemplate(
    input_variables=["checked_assertions", "question"],
    template=_REVISED_ANSWER_TEMPLATE,
)
