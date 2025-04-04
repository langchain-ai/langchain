"""An agent designed to hold a conversation in addition to using tools."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from langchain_core._api import deprecated
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool
from pydantic import Field

from langchain._api.deprecation import AGENT_DEPRECATION_WARNING
from langchain.agents.agent import Agent, AgentOutputParser
from langchain.agents.agent_types import AgentType
from langchain.agents.conversational.output_parser import ConvoOutputParser
from langchain.agents.conversational.prompt import FORMAT_INSTRUCTIONS, PREFIX, SUFFIX
from langchain.agents.utils import validate_tools_single_input
from langchain.chains import LLMChain


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
    def _get_default_output_parser(
        cls, ai_prefix: str = "AI", **kwargs: Any
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
        input_variables: Optional[List[str]] = None,
    ) -> PromptTemplate:
        """Create prompt in the style of the zero-shot agent.

        Args:
            tools: List of tools the agent will have access to, used to format the
                prompt.
            prefix: String to put before the list of tools. Defaults to PREFIX.
            suffix: String to put after the list of tools. Defaults to SUFFIX.
            format_instructions: Instructions on how to use the tools. Defaults to
                FORMAT_INSTRUCTIONS
            ai_prefix: String to use before AI output. Defaults to "AI".
            human_prefix: String to use before human output.
                Defaults to "Human".
            input_variables: List of input variables the final prompt will expect.
                Defaults to ["input", "chat_history", "agent_scratchpad"].

        Returns:
            A PromptTemplate with the template assembled from the pieces here.
        """
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
        callback_manager: Optional[BaseCallbackManager] = None,
        output_parser: Optional[AgentOutputParser] = None,
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        ai_prefix: str = "AI",
        human_prefix: str = "Human",
        input_variables: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Agent:
        """Construct an agent from an LLM and tools.

        Args:
            llm: The language model to use.
            tools: A list of tools to use.
            callback_manager: The callback manager to use. Default is None.
            output_parser: The output parser to use. Default is None.
            prefix: The prefix to use in the prompt. Default is PREFIX.
            suffix: The suffix to use in the prompt. Default is SUFFIX.
            format_instructions: The format instructions to use.
                Default is FORMAT_INSTRUCTIONS.
            ai_prefix: The prefix to use before AI output. Default is "AI".
            human_prefix: The prefix to use before human output.
                Default is "Human".
            input_variables: The input variables to use. Default is None.
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
        llm_chain = LLMChain(  # type: ignore[misc]
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        _output_parser = output_parser or cls._get_default_output_parser(
            ai_prefix=ai_prefix
        )
        return cls(
            llm_chain=llm_chain,
            allowed_tools=tool_names,
            ai_prefix=ai_prefix,
            output_parser=_output_parser,
            **kwargs,
        )
