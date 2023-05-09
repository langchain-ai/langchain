import re

from langchain.agents.plan_and_execute.planners.base import LLMPlanner
from langchain.agents.plan_and_execute.schema import Plan, PlanOutputParser
from langchain.base_language import BaseLanguageModel
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage

SYSTEM_PROMPT = (
    "Let's first understand the problem and devise a plan to solve the problem."
    " Please output the plan starting with the header 'Plan:' "
    "and then followed by a numbered list of steps. "
    "Please make the plan the minimum number of steps required "
    "to accurately complete the task. If the task is a question, "
    "the final step should almost always be 'Given the above steps taken, "
    "please respond to the users original question'. "
    "At the end of your plan, say '<END_OF_PLAN>'"
)


class PlanOP(PlanOutputParser):
    def parse(self, text):
        return Plan(steps=re.split("\n\d+\. ", text)[1:])


def load_chat_planner(llm: BaseLanguageModel) -> LLMPlanner:
    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )
    llm_chain_1 = LLMChain(llm=llm, prompt=prompt_template)
    return LLMPlanner(
        llm_chain=llm_chain_1, output_parser=PlanOP(), stop=["<END_OF_PLAN>"]
    )
