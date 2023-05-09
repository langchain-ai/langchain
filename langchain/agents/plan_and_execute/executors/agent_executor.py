import re
from typing import Any, List, Optional, Sequence, Tuple

from pydantic import Field

from langchain.agents.agent import Agent, AgentExecutor, AgentOutputParser
from langchain.agents.plan_and_execute.executors.base import ChainExecutor
from langchain.agents.structured_chat.base import StructuredChatAgent
from langchain.agents.structured_chat.output_parser import (
    StructuredChatOutputParserWithRetries,
)
from langchain.agents.structured_chat.prompt import FORMAT_INSTRUCTIONS, PREFIX, SUFFIX
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import AgentAction
from langchain.tools import BaseTool


def load_agent_executor(
    llm: BaseLanguageModel, tools: List[BaseTool], verbose: bool = False
) -> ChainExecutor:
    agent = StructuredChatAgent.from_llm_and_tools(
        llm,
        tools,
        human_message_template="Previous steps: {previous_steps}\n\nCurrent objective: {current_step}\n\n{agent_scratchpad}",
        input_variables=["previous_steps", "current_step", "agent_scratchpad"],
    )
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=verbose
    )
    return ChainExecutor(chain=agent_executor)
