"""Attempt to implement MRKL systems as described in arxiv.org/pdf/2205.00445.pdf."""
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

from pydantic import BaseModel, Extra

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.mrkl.prompt import BASE_TEMPLATE
from langchain.chains.router import LLMRouterChain
from langchain.input import ChainedInput, get_color_mapping
from langchain.llms.base import LLM
from langchain.prompts import Prompt

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
    if ps[-1].startswith(FINAL_ANSWER_ACTION):
        directive = ps[-1][len(FINAL_ANSWER_ACTION) :]
        return FINAL_ANSWER_ACTION, directive
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


class MRKLRouterChain(LLMRouterChain):
    """Router for the MRKL chain."""

    def __init__(self, llm: LLM, chain_configs: List[ChainConfig], **kwargs: Any):
        """Initialize with an LLM and the chain configs it has access to."""
        tools = "\n".join(
            [f"{c.action_name}: {c.action_description}" for c in chain_configs]
        )
        tool_names = ", ".join([chain.action_name for chain in chain_configs])
        template = BASE_TEMPLATE.format(tools=tools, tool_names=tool_names)
        prompt = Prompt(template=template, input_variables=["input"])
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        stops = ["\nObservation"]
        super().__init__(llm_chain=llm_chain, stops=stops, **kwargs)

    def _extract_action_and_input(self, text: str) -> Optional[Tuple[str, str]]:
        return get_action_and_input(text)


class MRKLChain(Chain, BaseModel):
    """Chain that implements the MRKL system.

    Example:
        .. code-block:: python

            from langchain import OpenAI, Prompt, MRKLChain
            from langchain.chains.mrkl.base import ChainConfig
            llm = OpenAI(temperature=0)
            prompt = Prompt(...)
            action_to_chain_map = {...}
            mrkl = MRKLChain(
                llm=llm,
                prompt=prompt,
                action_to_chain_map=action_to_chain_map
            )
    """

    llm: LLM
    """LLM wrapper to use as router."""
    chain_configs: List[ChainConfig]
    """Chain configs this chain has access to."""
    action_to_chain_map: Dict[str, Callable]
    """Mapping from action name to chain to execute."""
    input_key: str = "question"  #: :meta private:
    output_key: str = "answer"  #: :meta private:

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
        action_to_chain_map = {chain.action_name: chain.action for chain in chains}
        return cls(
            llm=llm,
            chain_configs=chains,
            action_to_chain_map=action_to_chain_map,
            **kwargs,
        )

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Expect output key.

        :meta private:
        """
        return [self.output_key]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        router_chain = MRKLRouterChain(self.llm, self.chain_configs)
        chained_input = ChainedInput(
            f"{inputs[self.input_key]}\nThought:", verbose=self.verbose
        )
        color_mapping = get_color_mapping(
            list(self.action_to_chain_map.keys()), excluded_colors=["green"]
        )
        while True:
            action, action_input, thought = router_chain.get_action_and_input(
                chained_input.input
            )
            chained_input.add(thought, color="green")
            if action == FINAL_ANSWER_ACTION:
                return {self.output_key: action_input}
            chain = self.action_to_chain_map[action]
            ca = chain(action_input)
            chained_input.add("\nObservation: ")
            chained_input.add(ca, color=color_mapping[action])
            chained_input.add("\nThought:")
