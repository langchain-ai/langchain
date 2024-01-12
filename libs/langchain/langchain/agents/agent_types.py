"""Module definitions of agent types together with corresponding agents."""
from enum import Enum

from langchain_core._api import deprecated


@deprecated(
    "0.1.0",
    alternative=(
        "Use new agent constructor methods like create_react_agent, create_json_agent, "
        "create_structured_chat_agent, etc."
    ),
    removal="0.2.0",
)
class AgentType(str, Enum):
    """An enum for agent types.

    See documentation: https://python.langchain.com/docs/modules/agents/agent_types/
    """

    ZERO_SHOT_REACT_DESCRIPTION = "zero-shot-react-description"
    """A zero shot agent that does a reasoning step before acting."""

    REACT_DOCSTORE = "react-docstore"
    """A zero shot agent that does a reasoning step before acting.
    
    This agent has access to a document store that allows it to look up 
    relevant information to answering the question.
    """

    SELF_ASK_WITH_SEARCH = "self-ask-with-search"
    """An agent that breaks down a complex question into a series of simpler questions.
    
    This agent uses a search tool to look up answers to the simpler questions
    in order to answer the original complex question.
    """
    CONVERSATIONAL_REACT_DESCRIPTION = "conversational-react-description"
    CHAT_ZERO_SHOT_REACT_DESCRIPTION = "chat-zero-shot-react-description"
    """A zero shot agent that does a reasoning step before acting.
    
    This agent is designed to be used in conjunction 
    """

    CHAT_CONVERSATIONAL_REACT_DESCRIPTION = "chat-conversational-react-description"

    STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION = (
        "structured-chat-zero-shot-react-description"
    )
    """An zero-shot react agent optimized for chat models.
    
    This agent is capable of invoking tools that have multiple inputs.
    """

    OPENAI_FUNCTIONS = "openai-functions"
    """An agent optimized for using open AI functions."""

    OPENAI_MULTI_FUNCTIONS = "openai-multi-functions"
