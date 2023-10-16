import re
from typing import List, Tuple

from langchain.agents import ConversationalChatAgent
from langchain.agents.conversational_chat.prompt import TEMPLATE_TOOL_RESPONSE
from langchain.schema import AgentAction, AIMessage, BaseMessage, HumanMessage

CONTEXT_PATTERN = re.compile(r"^CONTEXT:")


class ConversationalChatContextAgent(ConversationalChatAgent):
    """
    An agent designed to hold a conversation in addition to using tools.
    This agent can ask for context from the user. To ask for context,
    tools have to return a prefix 'CONTEXT:' followed by the context question.
    """

    @property
    def _agent_type(self) -> str:
        raise NotImplementedError

    def _construct_scratchpad(
        self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> List[BaseMessage]:
        """Construct the scratchpad that lets the agent continue its thought process."""
        thoughts: List[BaseMessage] = []
        for action, observation in intermediate_steps:
            thoughts.append(AIMessage(content=action.log))
            if observation and re.match(CONTEXT_PATTERN, observation):
                # remove the context_prefix from the observation
                human_message = HumanMessage(
                    content=re.sub(CONTEXT_PATTERN, "", observation)
                )
            else:
                human_message = HumanMessage(
                    content=TEMPLATE_TOOL_RESPONSE.format(observation=observation)
                )
            thoughts.append(human_message)
        return thoughts
