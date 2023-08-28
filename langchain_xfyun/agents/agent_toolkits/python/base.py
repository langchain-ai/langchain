"""Python agent."""

from typing import Any, Dict, Optional

from langchain_xfyun.agents.agent import AgentExecutor, BaseSingleActionAgent
from langchain_xfyun.agents.agent_toolkits.python.prompt import PREFIX
from langchain_xfyun.agents.mrkl.base import ZeroShotAgent
from langchain_xfyun.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain_xfyun.agents.types import AgentType
from langchain_xfyun.callbacks.base import BaseCallbackManager
from langchain_xfyun.chains.llm import LLMChain
from langchain_xfyun.schema.language_model import BaseLanguageModel
from langchain_xfyun.schema.messages import SystemMessage
from langchain_xfyun.tools.python.tool import PythonREPLTool


def create_python_agent(
    llm: BaseLanguageModel,
    tool: PythonREPLTool,
    agent_type: AgentType = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    callback_manager: Optional[BaseCallbackManager] = None,
    verbose: bool = False,
    prefix: str = PREFIX,
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Dict[str, Any],
) -> AgentExecutor:
    """Construct a python agent from an LLM and tool."""
    tools = [tool]
    agent: BaseSingleActionAgent

    if agent_type == AgentType.ZERO_SHOT_REACT_DESCRIPTION:
        prompt = ZeroShotAgent.create_prompt(tools, prefix=prefix)
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names, **kwargs)
    elif agent_type == AgentType.OPENAI_FUNCTIONS:
        system_message = SystemMessage(content=prefix)
        _prompt = OpenAIFunctionsAgent.create_prompt(system_message=system_message)
        agent = OpenAIFunctionsAgent(
            llm=llm,
            prompt=_prompt,
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
