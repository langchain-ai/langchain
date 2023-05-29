from typing import List

from langchain.agents.agent import AgentExecutor
from langchain.agents.structured_chat.base import StructuredChatAgent
from langchain.base_language import BaseLanguageModel
from langchain.experimental.plan_and_execute.executors.base import ChainExecutor
from langchain.tools import BaseTool

HUMAN_MESSAGE_TEMPLATE = """Previous steps: {previous_steps}

Current objective: {current_step}

{agent_scratchpad}"""

TASK_SUFFIX = """{objective}

"""


def load_agent_executor(
    llm: BaseLanguageModel,
    tools: List[BaseTool],
    verbose: bool = False,
    include_task_in_prompt: bool = False,
) -> ChainExecutor:
    input_variables = ["previous_steps", "current_step", "agent_scratchpad"]
    TEMPLATE = HUMAN_MESSAGE_TEMPLATE

    if include_task_in_prompt:
        input_variables.append("objective")
        TEMPLATE = TASK_SUFFIX + TEMPLATE

    agent = StructuredChatAgent.from_llm_and_tools(
        llm,
        tools,
        human_message_template=TEMPLATE,
        input_variables=input_variables,
    )
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=verbose
    )
    return ChainExecutor(chain=agent_executor)
