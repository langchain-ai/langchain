"""Attempt to implement MRKL systems as described in arxiv.org/pdf/2205.00445.pdf."""
from typing import Any, Callable, List, NamedTuple, Optional, Tuple

from langchain.chains.llm import LLMChain
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.routing_chains.mrkl.prompt import BASE_TEMPLATE
from langchain.routing_chains.router import LLMRouter
from langchain.routing_chains.routing_chain import RoutingChain, ToolConfig

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


class MRKLRouterChain(LLMRouter):
    """Router for the MRKL chain."""

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "Observation: "

    @property
    def router_prefix(self) -> str:
        """Prefix to append the router call with."""
        return "Thought:"

    def __init__(self, llm: LLM, chain_configs: List[ChainConfig], **kwargs: Any):
        """Initialize with an LLM and the chain configs it has access to."""
        tools = "\n".join(
            [f"{c.action_name}: {c.action_description}" for c in chain_configs]
        )
        tool_names = ", ".join([chain.action_name for chain in chain_configs])
        template = BASE_TEMPLATE.format(tools=tools, tool_names=tool_names)
        prompt = PromptTemplate(template=template, input_variables=["input"])
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        stops = ["\nObservation"]
        super().__init__(llm_chain=llm_chain, stops=stops, **kwargs)

    def _extract_tool_and_input(self, text: str) -> Optional[Tuple[str, str]]:
        return get_action_and_input(text)


class MRKLChain(RoutingChain):
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
    ) -> "MRKLChain":
        """User friendly way to initialize the MRKL chain.

        This is intended to be an easy way to get up and running with the
        MRKL chain.

        Args:
            llm: The LLM to use as the router LLM.
            chains: The chains the MRKL system has access to.
            **kwargs: parameters to be passed to initialization.

        Returns:
            An initialized MRKL chain.

        Example:
            .. code-block:: python

                from langchain import LLMMathChain, OpenAI, SerpAPIChain, MRKLChain
                from langchain.chains.mrkl.base import ChainConfig
                llm = OpenAI(temperature=0)
                search = SerpAPIChain()
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
        router_chain = MRKLRouterChain(llm, chains)
        expert_configs = [
            ToolConfig(tool_name=c.action_name, tool=c.action) for c in chains
        ]
        return cls(router_chain=router_chain, expert_configs=expert_configs, **kwargs)
