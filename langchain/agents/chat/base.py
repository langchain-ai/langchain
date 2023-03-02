from typing import List, Optional, Sequence, Tuple

from langchain.agents.chat.prompt import FORMAT_INSTRUCTIONS, PREFIX, SUFFIX
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import AgentAction
from langchain.tools import BaseTool

FINAL_ANSWER_ACTION = "Final Answer:"


class ChatAgent(ZeroShotAgent):
    def _construct_scratchpad(
        self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> str:
        agent_scratchpad = super()._construct_scratchpad(intermediate_steps)
        if agent_scratchpad:
            return (
                f"This was your previous work "
                f"(but I haven't seen any of it! I only see what "
                f"you return as final answer):\n{agent_scratchpad}"
            )
        else:
            return agent_scratchpad

    def _extract_tool_and_input(self, text: str) -> Optional[Tuple[str, str]]:
        if FINAL_ANSWER_ACTION in text:
            return "Final Answer", text.split(FINAL_ANSWER_ACTION)[-1].strip()
        _, action, _ = text.split("```")
        import json

        foo = json.loads(action.strip())
        return foo["action"], foo["action_input"]

    @property
    def _stop(self) -> List[str]:
        return ["Observation:"]

    @classmethod
    def create_prompt(
        cls,
        tools: Sequence[BaseTool],
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        input_variables: Optional[List[str]] = None,
    ) -> BasePromptTemplate:
        tool_strings = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
        tool_names = ", ".join([tool.name for tool in tools])
        format_instructions = format_instructions.format(tool_names=tool_names)
        template = "\n\n".join([prefix, tool_strings, format_instructions, suffix])
        messages = [
            ("user", PromptTemplate.from_template(template)),
            ("user", PromptTemplate.from_template("{input}\n\n{agent_scratchpad}")),
        ]
        return ChatPromptTemplate(
            input_variables=["input", "agent_scratchpad"], messages=messages
        )

    @property
    def _agent_type(self) -> str:
        raise ValueError
