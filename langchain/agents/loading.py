"""Load agent."""
from typing import Any, List

from langchain.agents.agent import AgentWithTools
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.react.base import ReActDocstoreAgent
from langchain.agents.self_ask_with_search.base import SelfAskWithSearchAgent
from langchain.agents.tools import Tool
from langchain.llms.base import LLM

AGENT_TO_CLASS = {
    "zero-shot-react-description": ZeroShotAgent,
    "react-docstore": ReActDocstoreAgent,
    "self-ask-with-search": SelfAskWithSearchAgent,
}


def initialize_agent(
    tools: List[Tool],
    llm: LLM,
    agent: str = "zero-shot-react-description",
    **kwargs: Any,
) -> AgentWithTools:
    """Load agent given tools and LLM.

    Args:
        tools: List of tools this agent has access to.
        llm: Language model to use as the agent.
        agent: The agent to use. Valid options are:
            `zero-shot-react-description`, `react-docstore`, `self-ask-with-search`.
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
    agent_obj = agent_cls.from_llm_and_tools(llm, tools)
    return AgentWithTools(agent=agent_obj, tools=tools, **kwargs)
