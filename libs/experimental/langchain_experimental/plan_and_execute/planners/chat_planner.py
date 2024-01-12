import re

from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import SystemMessage

from langchain_experimental.plan_and_execute.planners.base import LLMPlanner
from langchain_experimental.plan_and_execute.schema import (
    Plan,
    PlanOutputParser,
    Step,
)

SYSTEM_PROMPT = (
    "Давай сначала поймем проблему и разработаем план для ее решения."
    " Пожалуйста, выведи план, начиная с заголовка 'План:', "
    "а затем следуя нумерованным шагам. "
    "План должен содержать минимальное количество шагов, "
    "необходимых для точного выполнения задачи. Если задача представляет собой вопрос, "
    "то последним шагом почти всегда должно быть 'Учитывая приведенные выше шаги, "
    "пожалуйста, ответь на исходный вопрос пользователя'. "
    "В конце своего плана скажи '<END_OF_PLAN>'"
)


class PlanningOutputParser(PlanOutputParser):
    """Парсер выходных данных планирования."""

    def parse(self, text: str) -> Plan:
        steps = [Step(value=v) for v in re.split("\n\s*\d+\. ", text)[1:]]
        return Plan(steps=steps)


def load_chat_planner(
    llm: BaseLanguageModel, system_prompt: str = SYSTEM_PROMPT
) -> LLMPlanner:
    """
    Загрузить планировщик чата.

    Args:
        llm: Языковая модель.
        system_prompt: Системный запрос.

    Returns:
        LLMPlanner
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    return LLMPlanner(
        llm_chain=llm_chain,
        output_parser=PlanningOutputParser(),
        stop=["<END_OF_PLAN>"],
    )
