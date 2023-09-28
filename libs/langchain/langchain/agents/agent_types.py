"""Module definitions of agent types together with corresponding agents."""
from enum import Enum
from typing import Dict, Type, Union

from langchain.agents.agent import BaseSingleActionAgent
from langchain.agents.openai_functions_multi_agent.base import OpenAIMultiFunctionsAgent


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


AGENT_TYPE = Union[Type[BaseSingleActionAgent], Type[OpenAIMultiFunctionsAgent]]


def get_agent_to_class() -> Dict[AgentType, AGENT_TYPE]:
    """Get an agent to class mapping."""
    from langchain.agents.chat.base import ChatAgent
    from langchain.agents.conversational.base import ConversationalAgent
    from langchain.agents.conversational_chat.base import ConversationalChatAgent
    from langchain.agents.mrkl.base import ZeroShotAgent
    from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
    from langchain.agents.openai_functions_multi_agent.base import (
        OpenAIMultiFunctionsAgent,
    )
    from langchain.agents.react.base import ReActDocstoreAgent
    from langchain.agents.self_ask_with_search.base import SelfAskWithSearchAgent
    from langchain.agents.structured_chat.base import StructuredChatAgent

    return {
        AgentType.ZERO_SHOT_REACT_DESCRIPTION: ZeroShotAgent,
        AgentType.REACT_DOCSTORE: ReActDocstoreAgent,
        AgentType.SELF_ASK_WITH_SEARCH: SelfAskWithSearchAgent,
        AgentType.CONVERSATIONAL_REACT_DESCRIPTION: ConversationalAgent,
        AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION: ChatAgent,
        AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION: ConversationalChatAgent,
        AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION: StructuredChatAgent,
        AgentType.OPENAI_FUNCTIONS: OpenAIFunctionsAgent,
        AgentType.OPENAI_MULTI_FUNCTIONS: OpenAIMultiFunctionsAgent,
    }


AGENT_TO_CLASS = get_agent_to_class()
