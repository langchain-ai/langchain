"""Attempt to implement MRKL systems as described in arxiv.org/pdf/2205.00445.pdf."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, NamedTuple, Optional

from langchain_core._api import deprecated
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool, Tool
from langchain_core.tools.render import render_text_description
from pydantic import Field

from langchain._api.deprecation import AGENT_DEPRECATION_WARNING
from langchain.agents.agent import Agent, AgentExecutor, AgentOutputParser
from langchain.agents.agent_types import AgentType
from langchain.agents.mrkl.output_parser import MRKLOutputParser
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS, PREFIX, SUFFIX
from langchain.agents.utils import validate_tools_single_input
from langchain.chains import LLMChain


class ChainConfig(NamedTuple):
    """Configuration for a chain to use in MRKL system.

    Parameters:
        action_name: Name of the action.
        action: Action function to call.
        action_description: Description of the action.
    """

    action_name: str
    action: Callable
    action_description: str


@deprecated(
    "0.1.0",
    message=AGENT_DEPRECATION_WARNING,
    removal="1.0",
)
class ZeroShotAgent(Agent):
    """Agent for the MRKL chain.

    Parameters:
        output_parser: Output parser for the agent.
    """

    output_parser: AgentOutputParser = Field(default_factory=MRKLOutputParser)

    @classmethod
    def _get_default_output_parser(cls, **kwargs: Any) -> AgentOutputParser:
        return MRKLOutputParser()

    @property
    def _agent_type(self) -> str:
        """Return Identifier of agent type."""
        return AgentType.ZERO_SHOT_REACT_DESCRIPTION

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
        input_variables: Optional[list[str]] = None,
    ) -> PromptTemplate:
        """Create prompt in the style of the zero shot agent.

        Args:
            tools: List of tools the agent will have access to, used to format the
                prompt.
            prefix: String to put before the list of tools. Defaults to PREFIX.
            suffix: String to put after the list of tools. Defaults to SUFFIX.
            format_instructions: Instructions on how to use the tools.
                Defaults to FORMAT_INSTRUCTIONS
            input_variables: List of input variables the final prompt will expect.
                Defaults to None.

        Returns:
            A PromptTemplate with the template assembled from the pieces here.
        """
        tool_strings = render_text_description(list(tools))
        tool_names = ", ".join([tool.name for tool in tools])
        format_instructions = format_instructions.format(tool_names=tool_names)
        template = "\n\n".join([prefix, tool_strings, format_instructions, suffix])
        if input_variables:
            return PromptTemplate(template=template, input_variables=input_variables)
        return PromptTemplate.from_template(template)

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
        input_variables: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Agent:
        """Construct an agent from an LLM and tools.

        Args:
            llm: The LLM to use as the agent LLM.
            tools: The tools to use.
            callback_manager: The callback manager to use. Defaults to None.
            output_parser: The output parser to use. Defaults to None.
            prefix: The prefix to use. Defaults to PREFIX.
            suffix: The suffix to use. Defaults to SUFFIX.
            format_instructions: The format instructions to use.
                Defaults to FORMAT_INSTRUCTIONS.
            input_variables: The input variables to use. Defaults to None.
            kwargs: Additional parameters to pass to the agent.
        """
        cls._validate_tools(tools)
        prompt = cls.create_prompt(
            tools,
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
        _output_parser = output_parser or cls._get_default_output_parser()
        return cls(
            llm_chain=llm_chain,
            allowed_tools=tool_names,
            output_parser=_output_parser,
            **kwargs,
        )

    @classmethod
    def _validate_tools(cls, tools: Sequence[BaseTool]) -> None:
        validate_tools_single_input(cls.__name__, tools)
        if len(tools) == 0:
            raise ValueError(
                f"Got no tools for {cls.__name__}. At least one tool must be provided."
            )
        for tool in tools:
            if tool.description is None:
                raise ValueError(
                    f"Got a tool {tool.name} without a description. For this agent, "
                    f"a description must always be provided."
                )
        super()._validate_tools(tools)


@deprecated(
    "0.1.0",
    message=AGENT_DEPRECATION_WARNING,
    removal="1.0",
)
class MRKLChain(AgentExecutor):
    """Chain that implements the MRKL system."""

    @classmethod
    def from_chains(
        cls, llm: BaseLanguageModel, chains: list[ChainConfig], **kwargs: Any
    ) -> AgentExecutor:
        """User-friendly way to initialize the MRKL chain.

        This is intended to be an easy way to get up and running with the
        MRKL chain.

        Args:
            llm: The LLM to use as the agent LLM.
            chains: The chains the MRKL system has access to.
            **kwargs: parameters to be passed to initialization.

        Returns:
            An initialized MRKL chain.
        """
        tools = [
            Tool(
                name=c.action_name,
                func=c.action,
                description=c.action_description,
            )
            for c in chains
        ]
        agent = ZeroShotAgent.from_llm_and_tools(llm, tools)
        return cls(agent=agent, tools=tools, **kwargs)
