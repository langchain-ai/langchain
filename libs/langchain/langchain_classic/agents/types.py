from langchain_classic.agents.agent import BaseSingleActionAgent
from langchain_classic.agents.agent_types import AgentType
from langchain_classic.agents.chat.base import ChatAgent
from langchain_classic.agents.conversational.base import ConversationalAgent
from langchain_classic.agents.conversational_chat.base import ConversationalChatAgent
from langchain_classic.agents.mrkl.base import ZeroShotAgent
from langchain_classic.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain_classic.agents.openai_functions_multi_agent.base import (
    OpenAIMultiFunctionsAgent,
)
from langchain_classic.agents.react.base import ReActDocstoreAgent
from langchain_classic.agents.self_ask_with_search.base import SelfAskWithSearchAgent
from langchain_classic.agents.structured_chat.base import StructuredChatAgent

AGENT_TYPE = type[BaseSingleActionAgent] | type[OpenAIMultiFunctionsAgent]

AGENT_TO_CLASS: dict[AgentType, AGENT_TYPE] = {
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
