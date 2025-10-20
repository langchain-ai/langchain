"""An agent designed to hold a conversation in addition to using tools."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core._api import deprecated
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool
from pydantic import Field
from typing_extensions import override

from langchain_classic._api.deprecation import AGENT_DEPRECATION_WARNING
from langchain_classic.agents.agent import Agent, AgentOutputParser
from langchain_classic.agents.agent_types import AgentType
from langchain_classic.agents.conversational.output_parser import ConvoOutputParser
from langchain_classic.agents.conversational.prompt import (
    FORMAT_INSTRUCTIONS,
    PREFIX,
    SUFFIX,
)
from langchain_classic.agents.utils import validate_tools_single_input
from langchain_classic.chains import LLMChain


@deprecated(
    "0.1.0",
    message=AGENT_DEPRECATION_WARNING,
    removal="1.0",
)
class ConversationalAgent(Agent):
    """An agent that holds a conversation in addition to using tools."""

    ai_prefix: str = "AI"
    """Prefix to use before AI output."""
    output_parser: AgentOutputParser = Field(default_factory=ConvoOutputParser)
    """Output parser for the agent."""

    @classmethod
    @override
    def _get_default_output_parser(
        cls,
        ai_prefix: str = "AI",
        **kwargs: Any,
    ) -> AgentOutputParser:
        return ConvoOutputParser(ai_prefix=ai_prefix)

    @property
    def _agent_type(self) -> str:
        """Return Identifier of agent type."""
        return AgentType.CONVERSATIONAL_REACT_DESCRIPTION

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with.

        Returns:
            "Observation: "
        """
        return "Observation: "

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with.

        Returns:
            "Thought: "
        """
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
        input_variables: list[str] | None = None,
    ) -> PromptTemplate:
        """Create prompt in the style of the zero-shot agent.

        Args:
            tools: List of tools the agent will have access to, used to format the
                prompt.
            prefix: String to put before the list of tools.
            suffix: String to put after the list of tools.
            format_instructions: Instructions on how to use the tools.
            ai_prefix: String to use before AI output.
            human_prefix: String to use before human output.
            input_variables: List of input variables the final prompt will expect.
                Defaults to `["input", "chat_history", "agent_scratchpad"]`.

        Returns:
            A PromptTemplate with the template assembled from the pieces here.
        """
        tool_strings = "\n".join(
            [f"> {tool.name}: {tool.description}" for tool in tools],
        )
        tool_names = ", ".join([tool.name for tool in tools])
        format_instructions = format_instructions.format(
            tool_names=tool_names,
            ai_prefix=ai_prefix,
            human_prefix=human_prefix,
        )
        template = f"{prefix}\n\n{tool_strings}\n\n{format_instructions}\n\n{suffix}"
        if input_variables is None:
            input_variables = ["input", "chat_history", "agent_scratchpad"]
        return PromptTemplate(template=template, input_variables=input_variables)

    @classmethod
    def _validate_tools(cls, tools: Sequence[BaseTool]) -> None:
        super()._validate_tools(tools)
        validate_tools_single_input(cls.__name__, tools)

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callback_manager: BaseCallbackManager | None = None,
        output_parser: AgentOutputParser | None = None,
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        ai_prefix: str = "AI",
        human_prefix: str = "Human",
        input_variables: list[str] | None = None,
        **kwargs: Any,
    ) -> Agent:
        """Construct an agent from an LLM and tools.

        Args:
            llm: The language model to use.
            tools: A list of tools to use.
            callback_manager: The callback manager to use.
            output_parser: The output parser to use.
            prefix: The prefix to use in the prompt.
            suffix: The suffix to use in the prompt.
            format_instructions: The format instructions to use.
            ai_prefix: The prefix to use before AI output.
            human_prefix: The prefix to use before human output.
            input_variables: The input variables to use.
            **kwargs: Any additional keyword arguments to pass to the agent.

        Returns:
            An agent.
        """
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
        _output_parser = output_parser or cls._get_default_output_parser(
            ai_prefix=ai_prefix,
        )
        return cls(
            llm_chain=llm_chain,
            allowed_tools=tool_names,
            ai_prefix=ai_prefix,
            output_parser=_output_parser,
            **kwargs,
        )
