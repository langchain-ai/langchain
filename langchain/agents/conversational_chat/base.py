"""An agent designed to hold a conversation in addition to using tools."""
from __future__ import annotations

import json
from typing import Any, List, Optional, Sequence, Tuple

from langchain.agents.agent import Agent
from langchain.agents.conversational_chat.prompt import (
    PREFIX,
    SUFFIX,
)
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains import LLMChain
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.schema import BaseLanguageModel, AgentAction
from langchain.tools.base import BaseTool

template = """\n\nYou then took the following action:

```json
{{{{
    "action": "{tool}",
    "action_input": "{tool_input}"
}}}}
```

This yielded a response of:

```json
{{{{
    "response": "{observation}"
}}}}
```"""
def _format_step(action: AgentAction, observation: str):

    return template.format(tool=action.tool, tool_input=action.tool_input, observation=observation)


class ConversationalChatAgent(Agent):
    """An agent designed to hold a conversation in addition to using tools."""

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
        ai_prefix: str = "AI",
        human_prefix: str = "Human",
        input_variables: Optional[List[str]] = None,
    ) -> BasePromptTemplate:
        tool_strings = "\n".join(
            [f"> {tool.name}: {tool.description}" for tool in tools]
        )
        tool_names = ", ".join([tool.name for tool in tools])
        format_instructions = suffix.format(
            tool_names=tool_names, tools=tool_strings
        )

        if input_variables is None:
            input_variables = ["input", "chat_history", "agent_scratchpad"]
        messages = [
            SystemMessagePromptTemplate.from_template(prefix),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template(format_instructions)
        ]
        return ChatPromptTemplate(input_variables=input_variables, messages=messages)

    def _extract_tool_and_input(self, llm_output: str) -> Optional[Tuple[str, str]]:
        cleaned_output = llm_output.strip()
        if cleaned_output.startswith("```json"):
            cleaned_output = cleaned_output[len("```json"):]
        if cleaned_output.endswith("```"):
            cleaned_output = cleaned_output[:-len("```")]
        cleaned_output = cleaned_output.strip()
        try:
            response = json.loads(cleaned_output)
            return response["action"], response["action_input"]
        except Exception:
            raise ValueError(f"Could not parse LLM output: {llm_output}")

    def _construct_scratchpad(
        self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> str:
        """Construct the scratchpad that lets the agent continue its thought process."""
        thoughts = ""
        if intermediate_steps:
            thoughts += (
                "\n\nTool Use History:"
            )
            for action, observation in intermediate_steps:
                thoughts += _format_step(action, observation)
        return thoughts

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
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
