"""Python agent."""

from typing import Any, Optional

from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits.python.prompt import PREFIX
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.llms.base import BaseLLM
from langchain.tools.python.tool import PythonREPLTool


def create_python_agent(
    llm: BaseLLM,
    tool: PythonREPLTool,
    callback_manager: Optional[BaseCallbackManager] = None,
    verbose: bool = False,
    prefix: str = PREFIX,
    **kwargs: Any,
) -> AgentExecutor:
    """Construct a python agent from an LLM and tool."""
    tools = [tool]
    prompt = ZeroShotAgent.create_prompt(tools, prefix=prefix)
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        callback_manager=callback_manager,
    )
    tool_names = [tool.name for tool in tools]
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names, **kwargs)
    return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=verbose)
