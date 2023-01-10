"""Load agent."""
from typing import Any, List, Optional

from langchain.agents.agent import AgentExecutor
from langchain.agents.conversational.base import ConversationalAgent
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.react.base import ReActDocstoreAgent
from langchain.agents.self_ask_with_search.base import SelfAskWithSearchAgent
from langchain.agents.tools import Tool
from langchain.callbacks.base import BaseCallbackManager
from langchain.llms.base import BaseLLM

AGENT_TO_CLASS = {
    "zero-shot-react-description": ZeroShotAgent,
    "react-docstore": ReActDocstoreAgent,
    "self-ask-with-search": SelfAskWithSearchAgent,
    "conversational-react-description": ConversationalAgent,
}


def initialize_agent(
    tools: List[Tool],
    llm: BaseLLM,
    agent: str = "zero-shot-react-description",
    callback_manager: Optional[BaseCallbackManager] = None,
    **kwargs: Any,
) -> AgentExecutor:
    """Load agent given tools and LLM.

    Args:
        tools: List of tools this agent has access to.
        llm: Language model to use as the agent.
        agent: The agent to use. Valid options are:
            `zero-shot-react-description`
            `react-docstore`
            `self-ask-with-search`
            `conversational-react-description`.
        callback_manager: CallbackManager to use. Global callback manager is used if
            not provided. Defaults to None.
        **kwargs: Additional key word arguments to pass to the agent.

    Returns:
        An agent.
    """
    if agent not in AGENT_TO_CLASS:
        raise ValueError(
            f"Got unknown agent type: {agent}. "
            f"Valid types are: {AGENT_TO_CLASS.keys()}."
        )
    agent_cls = AGENT_TO_CLASS[agent]
    agent_obj = agent_cls.from_llm_and_tools(
        llm, tools, callback_manager=callback_manager
    )
    return AgentExecutor.from_agent_and_tools(
        agent=agent_obj,
        tools=tools,
        callback_manager=callback_manager,
        **kwargs,
    )
