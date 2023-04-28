from pydantic import BaseModel
from langchain.agents.agent import BaseSingleActionAgent
from typing import Type, List, Dict
from langchain.agents.agent_types import AgentType
from langchain.agents.chat.base import ChatAgent
from langchain.agents.conversational.base import ConversationalAgent
from langchain.agents.conversational_chat.base import ConversationalChatAgent
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.react.base import ReActDocstoreAgent
from langchain.agents.self_ask_with_search.base import SelfAskWithSearchAgent


class AgentTypeConfigBase:
    agent_type: AgentType
    execution_type: Type[BaseSingleActionAgent]


class ZeroShotReactDescriptionAgentTypeConfig(AgentTypeConfigBase):
    agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION
    execution_type = ZeroShotAgent

class ReactDocstoreAgentTypeConfig(AgentTypeConfigBase):
    agent_type = AgentType.REACT_DOCSTORE
    execution_type = ReActDocstoreAgent


class SelfAskWithSearchAgentTypeConfig(AgentTypeConfigBase):
    agent_type = AgentType.SELF_ASK_WITH_SEARCH
    execution_type = SelfAskWithSearchAgent


class ConversationalReactDescriptionAgentTypeConfig(AgentTypeConfigBase):
    agent_type = AgentType.CONVERSATIONAL_REACT_DESCRIPTION
    execution_type = ConversationalAgent


class ChatZeroShotReactDescriptionAgentTypeConfig(AgentTypeConfigBase):
    agent_type = AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION
    execution_type = ChatAgent


class ChatConversationalReactDescriptionAgentTypeConfig(AgentTypeConfigBase):
    agent_type = AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION
    execution_type = ConversationalChatAgent


agent_type_configs: List[Type[AgentTypeConfigBase]] = [
    ZeroShotReactDescriptionAgentTypeConfig,
    ReactDocstoreAgentTypeConfig,
    SelfAskWithSearchAgentTypeConfig,
    ConversationalReactDescriptionAgentTypeConfig,
    ChatZeroShotReactDescriptionAgentTypeConfig,
    ChatConversationalReactDescriptionAgentTypeConfig,
]
assert list(AgentType) == [c.agent_type for c in agent_type_configs], "AgentType doesn't match -AgentTypeConfig"

AGENT_TO_CLASS: Dict[AgentType, Type[BaseSingleActionAgent]] = {
    c.agent_type: c.execution_type for c in agent_type_configs
}