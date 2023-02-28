"""VectorStore agent."""
from typing import Any, List, Optional

from langchain.agents.agent import AgentExecutor
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.llms.base import BaseLLM
from langchain.tools.vectorstore.tool import VectorDBQATool


def create_vectorstore_agent(
    llm: BaseLLM,
    tools: List[VectorDBQATool],
    callback_manager: Optional[BaseCallbackManager] = None,
    verbose: bool = False,
    **kwargs: Any,
) -> AgentExecutor:
    """Construct a vectorstore agent from an LLM and tools."""
    prompt = ZeroShotAgent.create_prompt(tools)
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        callback_manager=callback_manager,
    )
    tool_names = [tool.name for tool in tools]
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names, **kwargs)
    return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=verbose)
