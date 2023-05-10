from typing import List

from langchain.agents.agent import AgentExecutor
from langchain.agents.structured_chat.base import StructuredChatAgent
from langchain.base_language import BaseLanguageModel
from langchain.experimental.plan_and_execute.executors.base import ChainExecutor
from langchain.tools import BaseTool

HUMAN_MESSAGE_TEMPLATE = """Previous steps: {previous_steps}

Current objective: {current_step}

{agent_scratchpad}"""


def load_agent_executor(
    llm: BaseLanguageModel, tools: List[BaseTool], verbose: bool = False
) -> ChainExecutor:
    agent = StructuredChatAgent.from_llm_and_tools(
        llm,
        tools,
        human_message_template=HUMAN_MESSAGE_TEMPLATE,
        input_variables=["previous_steps", "current_step", "agent_scratchpad"],
    )
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=verbose
    )
    return ChainExecutor(chain=agent_executor)
