"""Module definitions of agent types together with corresponding agents."""
from enum import Enum
from typing import Dict, Type, Union

from langchain.agents import (
    BaseSingleActionAgent,
    ConversationalAgent,
    ConversationalChatAgent,
    OpenAIFunctionsAgent,
    OpenAIMultiFunctionsAgent,
    StructuredChatAgent,
    ZeroShotAgent,
)
from langchain.agents.chat.base import ChatAgent
from langchain.agents.react.base import ReActDocstoreAgent
from langchain.agents.self_ask_with_search.base import SelfAskWithSearchAgent


class AgentType(str, Enum):
    """Enumerator with the Agent types."""

    ZERO_SHOT_REACT_DESCRIPTION = "zero-shot-react-description"
    """A zero shot agent that does a reasoning step before acting.
    
    Optimized for string based language models (not chat LLMs).
    """

    REACT_DOCSTORE = "react-docstore"
    """A zero shot agent that does a reasoning step before acting.
    
    This agent has access to a document store and can use it to look up relevant
    information to answering a question.
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
    
    This agent invokes tools using structured inputs rather than strings.
    """

    OPENAI_FUNCTIONS = "openai-functions"
    """An agent optimized for using open AI functions."""

    OPENAI_MULTI_FUNCTIONS = "openai-multi-functions"


AGENT_TYPE = Union[Type[BaseSingleActionAgent], Type[OpenAIMultiFunctionsAgent]]

AGENT_TO_CLASS: Dict[AgentType, AGENT_TYPE] = {
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
