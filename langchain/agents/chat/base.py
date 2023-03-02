from typing import List, Optional, Sequence

from langchain.agents.chat.prompt import FORMAT_INSTRUCTIONS, PREFIX, SUFFIX
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.tools import BaseTool


class ChatAgent(ZeroShotAgent):
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
            ("system", PromptTemplate.from_template(template)),
            ("user", PromptTemplate.from_template("{input}\n\n{agent_scratchpad}")),
        ]
        return ChatPromptTemplate(
            input_variables=["input", "agent_scratchpad"], messages=messages
        )

    @property
    def _agent_type(self) -> str:
        raise ValueError
