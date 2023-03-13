"""An agent designed to hold a conversation in addition to using tools."""
from __future__ import annotations

import json
from typing import Any, List, Optional, Sequence, Tuple

from langchain.agents.agent import Agent
from langchain.agents.conversational_chat.prompt import (
    FORMAT_INSTRUCTIONS,
    PREFIX,
    SUFFIX,
)
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains import LLMChain
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.schema import BaseLanguageModel
from langchain.tools.base import BaseTool


class ConversationalChatAgent(Agent):
    """An agent designed to hold a conversation in addition to using tools."""

    ai_prefix: str = "AI"

    @property
    def _agent_type(self) -> str:
        """Return Identifier of agent type."""
        return "chat-conversational-react-description"

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "Observation: "

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with."""
        return "Thought:"

    @classmethod
    def create_prompt(
        cls,
        tools: Sequence[BaseTool],
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        ai_prefix: str = "AI",
        human_prefix: str = "Human",
        input_variables: Optional[List[str]] = None,
    ) -> BasePromptTemplate:
        tool_strings = "\n".join(
            [f"> {tool.name}: {tool.description}" for tool in tools]
        )
        tool_names = ", ".join([tool.name for tool in tools])
        format_instructions = format_instructions.format(
            tool_names=tool_names, ai_prefix=ai_prefix, human_prefix=human_prefix
        )
        template = "\n\n".join([prefix, tool_strings, format_instructions, suffix])

        if input_variables is None:
            input_variables = ["input", "chat_history", "agent_scratchpad"]
        messages = [
            SystemMessagePromptTemplate.from_template(template),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}\n\n{agent_scratchpad}"),
        ]
        return ChatPromptTemplate(input_variables=input_variables, messages=messages)

    @property
    def finish_tool_name(self) -> str:
        """Name of the tool to use to finish the chain."""
        return self.ai_prefix

    def _extract_tool_and_input(self, llm_output: str) -> Optional[Tuple[str, str]]:
        try:
            response = json.loads(llm_output.strip())
            return response["action"], response["action_input"]
        except Exception:
            return self.ai_prefix, llm_output

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        ai_prefix: str = "AI",
        human_prefix: str = "Human",
        input_variables: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Agent:
        """Construct an agent from an LLM and tools."""
        cls._validate_tools(tools)
        prompt = cls.create_prompt(
            tools,
            ai_prefix=ai_prefix,
            human_prefix=human_prefix,
            prefix=prefix,
            suffix=suffix,
            format_instructions=format_instructions,
            input_variables=input_variables,
        )
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        return cls(
            llm_chain=llm_chain, allowed_tools=tool_names, ai_prefix=ai_prefix, **kwargs
        )
