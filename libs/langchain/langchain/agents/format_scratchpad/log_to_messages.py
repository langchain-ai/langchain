from typing import List, Tuple

from langchain_core.agents import AgentAction
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


def format_log_to_messages(
    intermediate_steps: List[Tuple[AgentAction, str]],
    template_tool_response: str = "{observation}",
) -> List[BaseMessage]:
    """Construct the scratchpad that lets the agent continue its thought process."""
    thoughts: List[BaseMessage] = []
    for action, observation in intermediate_steps:
        thoughts.append(AIMessage(content=action.log))
        human_message = HumanMessage(
            content=template_tool_response.format(observation=observation)
        )
        thoughts.append(human_message)
    return thoughts
