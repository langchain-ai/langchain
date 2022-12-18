"""Attempt to implement MRKL systems as described in arxiv.org/pdf/2205.00445.pdf."""
from __future__ import annotations

from typing import Any, Callable, List, NamedTuple, Optional, Tuple

from langchain.agents.agent import Agent, AgentWithTools
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS, PREFIX, SUFFIX
from langchain.agents.tools import Tool
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate

FINAL_ANSWER_ACTION = "Final Answer: "


class ChainConfig(NamedTuple):
    """Configuration for chain to use in MRKL system.

    Args:
        action_name: Name of the action.
        action: Action function to call.
        action_description: Description of the action.
    """

    action_name: str
    action: Callable
    action_description: str


def get_action_and_input(llm_output: str) -> Tuple[str, str]:
    """Parse out the action and input from the LLM output."""
    ps = [p for p in llm_output.split("\n") if p]
    if ps[-1].startswith("Final Answer"):
        directive = ps[-1][len(FINAL_ANSWER_ACTION) :]
        return "Final Answer", directive
    if not ps[-1].startswith("Action Input: "):
        raise ValueError(
            "The last line does not have an action input, "
            "something has gone terribly wrong."
        )
    if not ps[-2].startswith("Action: "):
        raise ValueError(
            "The second to last line does not have an action, "
            "something has gone terribly wrong."
        )
    action = ps[-2][len("Action: ") :]
    action_input = ps[-1][len("Action Input: ") :]
    return action, action_input.strip(" ").strip('"')


class ZeroShotAgent(Agent):
    """Agent for the MRKL chain."""

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
        tools: List[Tool],
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        input_variables: Optional[List[str]] = None,
    ) -> PromptTemplate:
        """Create prompt in the style of the zero shot agent.

        Args:
            tools: List of tools the agent will have access to, used to format the
                prompt.
            prefix: String to put before the list of tools.
            suffix: String to put after the list of tools.
            input_variables: List of input variables the final prompt will expect.

        Returns:
            A PromptTemplate with the template assembled from the pieces here.
        """
        tool_strings = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
        tool_names = ", ".join([tool.name for tool in tools])
        format_instructions = FORMAT_INSTRUCTIONS.format(tool_names=tool_names)
        template = "\n\n".join([prefix, tool_strings, format_instructions, suffix])
        if input_variables is None:
            input_variables = ["input", "agent_scratchpad"]
        return PromptTemplate(template=template, input_variables=input_variables)

    @classmethod
    def _validate_tools(cls, tools: List[Tool]) -> None:
        for tool in tools:
            if tool.description is None:
                raise ValueError(
                    f"Got a tool {tool.name} without a description. For this agent, "
                    f"a description must always be provided."
                )

    def _extract_tool_and_input(self, text: str) -> Optional[Tuple[str, str]]:
        return get_action_and_input(text)


class MRKLChain(AgentWithTools):
    """Chain that implements the MRKL system.

    Example:
        .. code-block:: python

            from langchain import OpenAI, MRKLChain
            from langchain.chains.mrkl.base import ChainConfig
            llm = OpenAI(temperature=0)
            prompt = PromptTemplate(...)
            chains = [...]
            mrkl = MRKLChain.from_chains(llm=llm, prompt=prompt)
    """

    @classmethod
    def from_chains(
        cls, llm: LLM, chains: List[ChainConfig], **kwargs: Any
    ) -> AgentWithTools:
        """User friendly way to initialize the MRKL chain.

        This is intended to be an easy way to get up and running with the
        MRKL chain.

        Args:
            llm: The LLM to use as the agent LLM.
            chains: The chains the MRKL system has access to.
            **kwargs: parameters to be passed to initialization.

        Returns:
            An initialized MRKL chain.

        Example:
            .. code-block:: python

                from langchain import LLMMathChain, OpenAI, SerpAPIWrapper, MRKLChain
                from langchain.chains.mrkl.base import ChainConfig
                llm = OpenAI(temperature=0)
                search = SerpAPIWrapper()
                llm_math_chain = LLMMathChain(llm=llm)
                chains = [
                    ChainConfig(
                        action_name = "Search",
                        action=search.search,
                        action_description="useful for searching"
                    ),
                    ChainConfig(
                        action_name="Calculator",
                        action=llm_math_chain.run,
                        action_description="useful for doing math"
                    )
                ]
                mrkl = MRKLChain.from_chains(llm, chains)
        """
        tools = [
            Tool(name=c.action_name, func=c.action, description=c.action_description)
            for c in chains
        ]
        agent = ZeroShotAgent.from_llm_and_tools(llm, tools)
        return cls(agent=agent, tools=tools, **kwargs)
