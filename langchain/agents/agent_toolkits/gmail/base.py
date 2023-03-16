from typing import Any, List, Optional

from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits.gmail.toolkit import GmailToolkit
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.llms.base import BaseLLM

from langchain.agents.agent_toolkits.gmail.prompt import GMAIL_PREFIX, GMAIL_SUFFIX


def create_gmail_agent(
    llm: BaseLLM,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: str = GMAIL_PREFIX,
    suffix: str = GMAIL_SUFFIX,
    format_instructions: str = FORMAT_INSTRUCTIONS,
    input_variables: Optional[List[str]] = None,
    verbose: bool = False,
    **kwargs: Any,
) -> AgentExecutor:
    """Construct a gmail agent from an LLM and tools."""
    toolkit = GmailToolkit(llm=llm, **kwargs)
    tools = toolkit.get_tools()
    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        format_instructions=format_instructions,
        input_variables=input_variables,
    )
    llm_chain = LLMChain(
        llm=llm, prompt=prompt, callback_manager=callback_manager, verbose=verbose
    )
    tool_names = [tool.name for tool in tools]
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names, **kwargs)
    return AgentExecutor.from_agent_and_tools(
        agent=agent, tools=toolkit.get_tools(), verbose=verbose
    )