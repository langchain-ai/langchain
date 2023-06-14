"""Json agent."""
from typing import Any, Dict, List, Optional

from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits.json.prompt import (
    JSON_PREFIX,
    JSON_SUFFIX,
    JSON_SUFFIX_FUNCTIONS,
)
from langchain.agents.agent_toolkits.json.toolkit import JsonToolkit
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.agents.types import AgentType
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.schema import SystemMessage


def create_json_agent(
    llm: BaseLanguageModel,
    toolkit: JsonToolkit,
    prefix: str = None,
    suffix: str = None,
    agent_type:AgentType = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    callback_manager: Optional[BaseCallbackManager] = None,
    format_instructions: str = FORMAT_INSTRUCTIONS,
    input_variables: Optional[List[str]] = None,
    verbose: bool = False,
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Dict[str, Any],
) -> AgentExecutor:
    """Construct a json agent from an LLM and tools."""
    tools = toolkit.get_tools()
    prefix = prefix or JSON_PREFIX

    if agent_type == AgentType.ZERO_SHOT_REACT_DESCRIPTION:
        suffix = suffix or JSON_SUFFIX
        prompt = ZeroShotAgent.create_prompt(
            tools,
            prefix=prefix,
            suffix=suffix,
            format_instructions=format_instructions,
            input_variables=input_variables,
        )
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names, **kwargs)
    elif agent_type == AgentType.OPENAI_FUNCTIONS:
        suffix = suffix or JSON_SUFFIX_FUNCTIONS

        system_message = SystemMessage(content=prefix+suffix)
        print(system_message)
        prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message
    )
        agent = OpenAIFunctionsAgent(
            llm=llm,
            prompt=prompt,
            tools=tools,
            callback_manager=callback_manager,
            **kwargs,
        )
    else:
        raise ValueError(f"Agent type {agent_type} not supported at the moment.")
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        callback_manager=callback_manager,
        verbose=verbose,
        **(agent_executor_kwargs or {}),
    )
