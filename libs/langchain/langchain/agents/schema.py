from typing import Any, Dict, List, Tuple

from langchain_core.agents import AgentAction
from langchain_core.prompts.chat import ChatPromptTemplate


class AgentScratchPadChatPromptTemplate(ChatPromptTemplate):
    """Chat prompt template for the agent scratchpad."""

    def _construct_agent_scratchpad(
        self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> str:
        if len(intermediate_steps) == 0:
            return ""
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        return (
            f"This was your previous work "
            f"(but I haven't seen any of it! I only see what "
            f"you return as final answer):\n{thoughts}"
        )

    def _merge_partial_and_user_variables(self, **kwargs: Any) -> Dict[str, Any]:
        intermediate_steps = kwargs.pop("intermediate_steps")
        kwargs["agent_scratchpad"] = self._construct_agent_scratchpad(
            intermediate_steps
        )
        return kwargs
